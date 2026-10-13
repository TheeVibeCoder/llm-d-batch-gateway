/*
Copyright 2026 The llm-d Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package worker

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	db "github.com/llm-d/llm-d-batch-gateway/internal/database/api"
	mockdb "github.com/llm-d/llm-d-batch-gateway/internal/database/mock"
	mockfiles "github.com/llm-d/llm-d-batch-gateway/internal/files_store/mock"
	"github.com/llm-d/llm-d-batch-gateway/internal/processor/config"
	"github.com/llm-d/llm-d-batch-gateway/internal/processor/metrics"
	"github.com/llm-d/llm-d-batch-gateway/internal/processor/pipeline"
	"github.com/llm-d/llm-d-batch-gateway/internal/shared/converter"
	"github.com/llm-d/llm-d-batch-gateway/internal/shared/openai"
	batch_types "github.com/llm-d/llm-d-batch-gateway/internal/shared/types"
	"github.com/llm-d/llm-d-batch-gateway/internal/util/clientset"
	ucom "github.com/llm-d/llm-d-batch-gateway/internal/util/com"
	"github.com/llm-d/llm-d-batch-gateway/pkg/clients/inference"
)

// durableJobEnv bundles the processor and mock stores of a durable (async
// dispatch) runJob test.
type durableJobEnv struct {
	p           *Processor
	dbClient    db.BatchProgressDBClient
	fileDB      db.FileDBClient
	filesClient *mockfiles.MockBatchFilesClient
	resumable   *mockdb.MockResumableStore
	pqClient    *mockdb.MockBatchPriorityQueueClient
	asyncClient *fakeAsyncInferenceClient
	jobItem     *db.BatchItem
	jobInfo     *batch_types.JobInfo
}

// durableInputLine builds one input line for the given model and custom ID.
func durableInputLine(model, customID string) string {
	return fmt.Sprintf(`{"custom_id":%q,"method":"POST","url":"/v1/chat/completions","body":{"model":%q,"messages":[{"role":"user","content":"hi"}]}}`+"\n", customID, model)
}

// seedDurableInput writes the batch input file for the given models into a
// fresh mock file store and records its file DB item, returning the store's
// root and the file DB client.
func seedDurableInput(t *testing.T, tenantID, inputFileID string, models []string) (string, db.FileDBClient) {
	t.Helper()

	filesRoot := t.TempDir()
	folderName, err := ucom.GetFolderNameByTenantID(tenantID)
	if err != nil {
		t.Fatalf("GetFolderNameByTenantID: %v", err)
	}

	var input bytes.Buffer
	for i, model := range models {
		input.WriteString(durableInputLine(model, fmt.Sprintf("req-%d", i+1)))
	}
	storageDir := filepath.Join(filesRoot, folderName)
	if err := os.MkdirAll(storageDir, 0o755); err != nil {
		t.Fatalf("MkdirAll storage dir: %v", err)
	}
	storageName := ucom.FileStorageName(inputFileID, "input.jsonl")
	if err := os.WriteFile(filepath.Join(storageDir, storageName), input.Bytes(), 0o644); err != nil {
		t.Fatalf("write input to mock storage: %v", err)
	}

	fileDBClient := newMockFileDBClient()
	fileItem, err := converter.FileToDBItem(&openai.FileObject{
		ID:       inputFileID,
		Filename: "input.jsonl",
		Purpose:  openai.FileObjectPurposeBatch,
		Object:   "file",
	}, tenantID, db.Tags{})
	if err != nil {
		t.Fatalf("FileToDBItem: %v", err)
	}
	if err := fileDBClient.DBStore(testLoggerCtx(t), fileItem); err != nil {
		t.Fatalf("DBStore file item: %v", err)
	}
	return filesRoot, fileDBClient
}

// setupDurableJob seeds a validating batch and wires a processor for the
// durable pipeline: an async resolver with "test-model" registered (every
// other model is rejected at ingestion), a mock resumable store seeded with
// the processor's claim, and manually-wired result broadcasters (Run wires
// them in production).
func setupDurableJob(t *testing.T, jobID string, models []string) *durableJobEnv {
	t.Helper()
	ctx := testLoggerCtx(t)

	const tenantID = "tenant-1"
	inputFileID := "file-input-" + jobID

	cfg := config.NewConfig()
	cfg.WorkDir = t.TempDir()
	filesRoot, fileDBClient := seedDurableInput(t, tenantID, inputFileID, models)

	dbClient := newMockBatchDBClient()
	jobItem := &db.BatchItem{
		BaseIndexes:  db.BaseIndexes{ID: jobID, TenantID: tenantID, Tags: db.Tags{}},
		BaseContents: db.BaseContents{Status: mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusValidating})},
		ProcessorID:  "test-processor",
		Epoch:        1,
		// The activation CTE flips the persisted resumable flag in production;
		// the mock stores cannot mirror that cross-store write, so seed the
		// post-activation value.
		Resumable: true,
	}
	if err := dbClient.DBStore(ctx, jobItem); err != nil {
		t.Fatalf("DBStore batch item: %v", err)
	}

	resumable := mockdb.NewMockResumableStore()
	resumable.SeedBatch(jobID, "test-processor", 1)

	asyncClient := &fakeAsyncInferenceClient{
		submittedCh: make(chan *inference.GenerateRequest),
		results:     make(chan *inference.GenerateResponse, 64),
	}
	resolver := inference.NewTestAsyncResolver(map[string]func() inference.AsyncInferenceClient{
		"test-model": func() inference.AsyncInferenceClient { return asyncClient },
	})

	pqClient := mockdb.NewMockBatchPriorityQueueClient()
	p := mustNewProcessor(t, cfg, &clientset.Clientset{
		BatchDB:         dbClient,
		BatchProgressDB: dbClient,
		FileDB:          fileDBClient,
		File:            mockfiles.NewMockBatchFilesClient(filesRoot),
		Event:           mockdb.NewMockBatchEventChannelClient(),
		Queue:           pqClient,
		AsyncInference:  resolver,
		Resumable:       resumable,
	})
	p.poller = NewPoller(pqClient, dbClient)

	bctx, bcancel := context.WithCancel(context.Background())
	p.broadcasters = newBroadcasterRegistry(resolver, testLogger(t))
	p.broadcasters.Run(bctx)
	t.Cleanup(func() {
		bcancel()
		p.broadcasters.Wait()
	})

	jobInfo := &batch_types.JobInfo{
		JobID:    jobID,
		TenantID: tenantID,
		BatchJob: &openai.Batch{
			ID: jobID,
			BatchSpec: openai.BatchSpec{
				InputFileID: inputFileID,
				Endpoint:    openai.EndpointChatCompletions,
			},
			BatchStatusInfo: openai.BatchStatusInfo{Status: openai.BatchStatusValidating},
		},
	}

	return &durableJobEnv{
		p:           p,
		dbClient:    dbClient,
		fileDB:      fileDBClient,
		filesClient: mockfiles.NewMockBatchFilesClient(filesRoot),
		resumable:   resumable,
		pqClient:    pqClient,
		asyncClient: asyncClient,
		jobItem:     jobItem,
		jobInfo:     jobInfo,
	}
}

func (e *durableJobEnv) runJob(t *testing.T) {
	t.Helper()
	if !e.p.acquire(context.Background()) {
		t.Fatalf("expected token acquire before runJob")
	}
	e.p.wg.Add(1)
	e.p.runJob(testLoggerCtx(t), &jobExecutionParams{
		updater: NewStatusUpdater(e.dbClient),
		jobItem: e.jobItem,
		jobInfo: e.jobInfo,
		task:    &db.BatchJobPriority{ID: e.jobItem.ID, SLO: time.Now().Add(1 * time.Hour)},
	})
}

// runResumedJobAsync launches runJob for a requeued resumable task and returns
// a channel closed when it returns. The token and logger context are taken on
// the caller's goroutine so a staging failure fails the test, not the spawned
// one.
func (e *durableJobEnv) runResumedJobAsync(t *testing.T, task *db.BatchJobPriority) <-chan struct{} {
	t.Helper()
	if !e.p.acquire(context.Background()) {
		t.Fatalf("expected token acquire before runJob")
	}
	ctx := testLoggerCtx(t)
	done := make(chan struct{})
	e.p.wg.Add(1)
	go func() {
		defer close(done)
		e.p.runJob(ctx, &jobExecutionParams{
			updater: NewStatusUpdater(e.dbClient),
			jobItem: e.jobItem,
			jobInfo: e.jobInfo,
			task:    task,
		})
	}()
	return done
}

// runResumedJob runs runJob synchronously with a requeued resumable task —
// the delivery shape of the unified dequeue.
func (e *durableJobEnv) runResumedJob(t *testing.T, task *db.BatchJobPriority) {
	t.Helper()
	<-e.runResumedJobAsync(t, task)
}

// pendingLineEntry builds the activation entry of one unfinished input line.
func pendingLineEntry(jobID, customID string, ordinal int64) db.ResumableActivationEntry {
	return db.ResumableActivationEntry{
		RequestID: stableBatchRequestID(jobID, customID),
		Ordinal:   ordinal,
		CustomID:  customID,
		Status:    string(db.ResumableEntryPending),
	}
}

// storeCheckpointedResult stores a previous incarnation's result object for
// one input line at its canonical key, returning its key, SHA-256, and size.
func storeCheckpointedResult(t *testing.T, filesClient *mockfiles.MockBatchFilesClient, tenantID, jobID, customID string) (string, string, int64) {
	t.Helper()
	reqID := stableBatchRequestID(jobID, customID)
	line, digest, size, err := pipeline.MarshalResultLine(&pipeline.ResultItem{
		RequestID: reqID,
		CustomID:  customID,
		Response: &batch_types.ResponseData{
			StatusCode: 200,
			RequestID:  reqID,
			Body:       map[string]any{"choices": []any{map[string]any{"text": "earlier-incarnation"}}},
		},
	})
	if err != nil {
		t.Fatalf("marshal seeded result for %s: %v", customID, err)
	}
	folder, err := ucom.GetFolderNameByTenantID(tenantID)
	if err != nil {
		t.Fatalf("GetFolderNameByTenantID: %v", err)
	}
	key := resumableResultKey(jobID, reqID)
	if _, err := filesClient.Store(testLoggerCtx(t), key, folder, 0, 0, bytes.NewReader(line)); err != nil {
		t.Fatalf("store seeded result object %s: %v", key, err)
	}
	return key, digest, size
}

// checkpointedLineEntry stores a previous incarnation's result object for one
// input line at its canonical key and returns the activation entry that
// checkpoints it.
func checkpointedLineEntry(t *testing.T, env *durableJobEnv, jobID, customID string, ordinal int64) db.ResumableActivationEntry {
	t.Helper()
	key, digest, size := storeCheckpointedResult(t, env.filesClient, env.jobInfo.TenantID, jobID, customID)
	return db.ResumableActivationEntry{
		RequestID:    stableBatchRequestID(jobID, customID),
		Ordinal:      ordinal,
		CustomID:     customID,
		Status:       string(db.ResumableEntryCompleted),
		ResultKey:    key,
		ResultBytes:  size,
		ResultSHA256: digest,
	}
}

// stageResumedClaim stages the mock resumable store at the state the unified
// dequeue leaves behind after re-claiming a requeued row: the processor owns
// the batch at the given epoch with the given (partially checkpointed) entry
// set. The requeue/claim bookkeeping itself is PG-tested — the mock cannot
// express a claim over an already-activated batch.
func (e *durableJobEnv) stageResumedClaim(t *testing.T, epoch int64, entries ...db.ResumableActivationEntry) {
	t.Helper()
	e.resumable.SeedBatch(e.jobItem.ID, e.jobItem.ProcessorID, epoch)
	err := e.resumable.ActivateResumableBatch(testLoggerCtx(t), &db.ResumableActivation{
		BatchID:     e.jobItem.ID,
		ProcessorID: e.jobItem.ProcessorID,
		Epoch:       epoch,
		LeaseTTL:    time.Minute,
		Entries:     entries,
	})
	if err != nil {
		t.Fatalf("stage resumed claim: %v", err)
	}
}

// durableJobStatus fetches the batch's persisted status info.
func durableJobStatus(t *testing.T, env *durableJobEnv) openai.BatchStatusInfo {
	t.Helper()
	items, _, _, err := env.dbClient.DBGet(context.Background(), &db.BatchQuery{BaseQuery: db.BaseQuery{IDs: []string{env.jobItem.ID}}}, true, 0, 1)
	if err != nil || len(items) != 1 {
		t.Fatalf("DBGet %s: err=%v len=%d", env.jobItem.ID, err, len(items))
	}
	var status openai.BatchStatusInfo
	if err := json.Unmarshal(items[0].Status, &status); err != nil {
		t.Fatalf("unmarshal status for %s: %v", env.jobItem.ID, err)
	}
	return status
}

// readArtifactLines retrieves and parses one finalization artifact.
func readArtifactLines(t *testing.T, env *durableJobEnv, key string) []outputLine {
	t.Helper()
	folderName, err := ucom.GetFolderNameByTenantID(env.jobInfo.TenantID)
	if err != nil {
		t.Fatalf("GetFolderNameByTenantID: %v", err)
	}
	reader, _, err := env.filesClient.Retrieve(context.Background(), key, folderName)
	if err != nil {
		t.Fatalf("retrieve artifact %s: %v", key, err)
	}
	defer reader.Close()
	data, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("read artifact %s: %v", key, err)
	}
	var lines []outputLine
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte{'\n'}) {
		if len(line) == 0 {
			continue
		}
		var ol outputLine
		if err := json.Unmarshal(line, &ol); err != nil {
			t.Fatalf("parse artifact line %s: %v", line, err)
		}
		lines = append(lines, ol)
	}
	return lines
}

// TestRunJob_Resumable_CompletesDurablePipeline verifies the durable
// orchestration end to end:
//
//	preProcessJob (ingestion rejection of the unregistered model)
//	→ activateResumable (rejected object PUT + fenced entry insert)
//	→ async dispatch with beforeSubmit fencing → result checkpointing
//	  (a 403 response is preserved as an output line, like the sync pipeline)
//	→ finalizeResumableJob (artifacts streamed from entry objects,
//	  deterministic file IDs, completed)
func TestRunJob_Resumable_CompletesDurablePipeline(t *testing.T) {
	const jobID = "job-resumable-happy"

	env := setupDurableJob(t, jobID, []string{"test-model", "test-model", "test-model", "missing-model"})

	// Answer the submitted requests: two successes around one preserved 403.
	go func() {
		for i := range 3 {
			req := <-env.asyncClient.submittedCh
			if i == 1 {
				env.asyncClient.deliverStatus(req.RequestID, http.StatusForbidden)
				continue
			}
			env.asyncClient.deliver(req.RequestID, map[string]any{"choices": []any{map[string]any{"text": "ok"}}})
		}
	}()

	env.runJob(t)

	reqID := func(customID string) string { return stableBatchRequestID(jobID, customID) }

	// Terminal state: completed, counts derived from the entry set, and
	// artifact file IDs derived from the batch.
	items, _, _, err := env.dbClient.DBGet(context.Background(), &db.BatchQuery{BaseQuery: db.BaseQuery{IDs: []string{jobID}}}, true, 0, 1)
	if err != nil || len(items) != 1 {
		t.Fatalf("DBGet: err=%v len=%d", err, len(items))
	}
	var status openai.BatchStatusInfo
	if err := json.Unmarshal(items[0].Status, &status); err != nil {
		t.Fatalf("unmarshal status: %v", err)
	}
	if status.Status != openai.BatchStatusCompleted {
		t.Fatalf("status = %s, want completed", status.Status)
	}
	wantCounts := openai.BatchRequestCounts{Total: 4, Completed: 2, Failed: 2}
	if status.RequestCounts != wantCounts {
		t.Fatalf("counts = %+v, want %+v", status.RequestCounts, wantCounts)
	}
	outFileID := stableFileID(jobID, metrics.FileTypeOutput)
	errFileID := stableFileID(jobID, metrics.FileTypeError)
	if status.OutputFileID == nil || *status.OutputFileID != outFileID {
		t.Fatalf("output_file_id = %v, want %s", status.OutputFileID, outFileID)
	}
	if status.ErrorFileID == nil || *status.ErrorFileID != errFileID {
		t.Fatalf("error_file_id = %v, want %s", status.ErrorFileID, errFileID)
	}

	// Only the registered-model requests were dispatched; the rejected line
	// never reached the queue.
	if got := env.asyncClient.submitCount(); got != 3 {
		t.Fatalf("submitted requests = %d, want 3", got)
	}

	// Entry set: two completed, two failed (the 403 and the rejected line).
	state, err := env.resumable.EntryResumeState(context.Background(), jobID)
	if err != nil {
		t.Fatalf("EntryResumeState: %v", err)
	}
	if state.Completed != 2 || state.Failed != 2 || state.Pending != 0 || state.InFlight != 0 {
		t.Fatalf("entry state = %+v, want 2 completed / 2 failed", state)
	}

	// Output artifact: the completed entries' objects plus the preserved 403,
	// in input order.
	outKey := ucom.FileStorageName(outFileID, jobOutputStorageName(jobID))
	outLines := readArtifactLines(t, env, outKey)
	if len(outLines) != 3 {
		t.Fatalf("output artifact lines = %d, want 3", len(outLines))
	}
	wantStatus := []int{http.StatusOK, http.StatusForbidden, http.StatusOK}
	for i, want := range []string{reqID("req-1"), reqID("req-2"), reqID("req-3")} {
		if outLines[i].ID != want {
			t.Fatalf("output line %d id = %s, want %s", i, outLines[i].ID, want)
		}
		if outLines[i].Error != nil || outLines[i].Response == nil || outLines[i].Response.StatusCode != wantStatus[i] {
			t.Fatalf("output line %d is not a %d response line: %+v", i, wantStatus[i], outLines[i])
		}
	}

	// Error artifact: the rejected entry's model_not_found object.
	errKey := ucom.FileStorageName(errFileID, jobErrorStorageName(jobID))
	errLines := readArtifactLines(t, env, errKey)
	if len(errLines) != 1 {
		t.Fatalf("error artifact lines = %d, want 1", len(errLines))
	}
	if errLines[0].ID != reqID("req-4") {
		t.Fatalf("error line id = %s, want %s", errLines[0].ID, reqID("req-4"))
	}
	if errLines[0].Error == nil || errLines[0].Error.Code != inference.ErrCodeModelNotFound {
		t.Fatalf("error line is not model_not_found: %+v", errLines[0])
	}

	// File records exist for both artifacts.
	for _, fileID := range []string{outFileID, errFileID} {
		files, _, _, err := env.fileDB.DBGet(context.Background(), &db.FileQuery{BaseQuery: db.BaseQuery{IDs: []string{fileID}}}, true, 0, 1)
		if err != nil || len(files) != 1 {
			t.Fatalf("file record %s: err=%v len=%d", fileID, err, len(files))
		}
	}
}

// TestRunJob_Resumable_ActivationConflict_LeavesJobValidating verifies that a
// fenced-out activation (another incarnation owns the batch) aborts without a
// terminal write and without inserting entries — the new owner owns the
// batch's terminal state.
func TestRunJob_Resumable_ActivationConflict_LeavesJobValidating(t *testing.T) {
	const jobID = "job-resumable-conflict"

	env := setupDurableJob(t, jobID, []string{"test-model"})
	// A foreign owner already claimed the batch at the same epoch.
	env.resumable.SeedBatch(jobID, "other-processor", 1)

	env.runJob(t)

	assertJobStatus(t, env.dbClient, jobID, openai.BatchStatusValidating)

	state, err := env.resumable.EntryResumeState(context.Background(), jobID)
	if err != nil {
		t.Fatalf("EntryResumeState: %v", err)
	}
	if state.Pending+state.InFlight+state.Completed+state.Failed != 0 {
		t.Fatalf("entries inserted despite activation conflict: %+v", state)
	}
}

// TestRunPollingLoop_ResumableRouting covers the polling loop's handling of a
// claimed requeued resumable task (task.Resumable): the loop hands it to
// runJob whose pickup routing terminalizes terminal-bound rows with partial
// artifacts, a processor without the durable pipeline skips it fail-closed,
// and the shutdown guard requeues it through the resumable store instead of
// the legacy enqueue.
func TestRunPollingLoop_ResumableRouting(t *testing.T) {
	t.Run("sync processor skips the task without a terminal write", func(t *testing.T) {
		ctx := testLoggerCtx(t)

		cfg := config.NewConfig()
		cfg.PollInterval = 5 * time.Millisecond
		cfg.NumWorkers = 1

		pq := &spyPQ{inner: mockdb.NewMockBatchPriorityQueueClient()}
		dbClient := newSpyBatchDB(newMockBatchDBClient())
		jobID := "job-resumable-sync-skip"

		jobItem := &db.BatchItem{
			BaseIndexes: db.BaseIndexes{
				ID:       jobID,
				TenantID: "tenantA",
				Tags:     db.Tags{"tenant": "tenantA"},
			},
			BaseContents: db.BaseContents{
				Spec:   mustJSON(t, openai.BatchSpec{InputFileID: "unused"}),
				Status: mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusInProgress}),
			},
		}
		if err := dbClient.DBStore(ctx, jobItem); err != nil {
			t.Fatalf("DBStore job item: %v", err)
		}
		if err := pq.PQEnqueue(ctx, &db.BatchJobPriority{
			ID:        jobID,
			SLO:       time.Now().Add(-1 * time.Second),
			Resumable: true,
		}); err != nil {
			t.Fatalf("PQEnqueue task: %v", err)
		}

		p := mustNewProcessor(t, cfg, &clientset.Clientset{BatchDB: dbClient, Queue: pq})

		runCtx, cancel := context.WithTimeout(ctx, 40*time.Millisecond)
		defer cancel()
		if err := p.runPollingLoop(runCtx, runCtx); err != nil {
			t.Fatalf("runPollingLoop: %v", err)
		}

		// Fail-closed: the lapsed SLO must not expire the batch and the
		// in-memory task is dropped, not re-enqueued.
		if calls := dbClient.StatusCalls(openai.BatchStatusExpired); calls != 0 {
			t.Fatalf("sync processor wrote expired status %d times for a resumable task", calls)
		}
		assertJobStatus(t, dbClient, jobID, openai.BatchStatusInProgress)
		if calls := pq.EnqueueCalls(); calls != 1 {
			t.Fatalf("enqueue calls = %d, want only the seed", calls)
		}
	})

	t.Run("durable pickup terminalizes a cancelling row with partial artifacts", func(t *testing.T) {
		const jobID = "job-resumable-poll-route"
		ctx := testLoggerCtx(t)

		env := setupDurableJob(t, jobID, []string{"test-model", "test-model"})

		// A requeued cancelling row whose previous incarnation checkpointed
		// one of two results. The poll loop must not terminalize it itself —
		// runJob's pickup routing owns the durable finalize: the completed
		// entry's object lands in the output artifact, the pending entry
		// becomes a not-executed cancelled line, and nothing is dispatched.
		env.jobItem.Spec = mustJSON(t, openai.BatchSpec{
			InputFileID: "file-input-" + jobID,
			Endpoint:    openai.EndpointChatCompletions,
		})
		env.jobItem.Status = mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusCancelling})
		env.jobItem.Epoch = 2
		if err := env.dbClient.DBStore(ctx, env.jobItem); err != nil {
			t.Fatalf("DBStore job item: %v", err)
		}
		env.stageResumedClaim(t, 2,
			checkpointedLineEntry(t, env, jobID, "req-1", 0),
			pendingLineEntry(jobID, "req-2", 1),
		)

		if err := env.pqClient.PQEnqueue(ctx, &db.BatchJobPriority{
			ID:        jobID,
			SLO:       time.Now().Add(1 * time.Hour),
			Epoch:     2,
			Resumable: true,
		}); err != nil {
			t.Fatalf("PQEnqueue task: %v", err)
		}

		runCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
		defer cancel()
		if err := env.p.runPollingLoop(runCtx, runCtx); err != nil {
			t.Fatalf("runPollingLoop: %v", err)
		}
		env.p.wg.Wait()

		status := durableJobStatus(t, env)
		if status.Status != openai.BatchStatusCancelled {
			t.Fatalf("status = %s, want cancelled", status.Status)
		}
		wantCounts := openai.BatchRequestCounts{Total: 2, Completed: 1, Failed: 1}
		if status.RequestCounts != wantCounts {
			t.Fatalf("counts = %+v, want %+v", status.RequestCounts, wantCounts)
		}

		// Output artifact: the checkpointed object of the completed entry.
		outLines := readArtifactLines(t, env,
			ucom.FileStorageName(stableFileID(jobID, metrics.FileTypeOutput), jobOutputStorageName(jobID)))
		if len(outLines) != 1 || outLines[0].ID != stableBatchRequestID(jobID, "req-1") {
			t.Fatalf("output artifact = %+v, want the checkpointed req-1 line", outLines)
		}

		// Error artifact: the pending entry as a not-executed cancelled line.
		errLines := readArtifactLines(t, env,
			ucom.FileStorageName(stableFileID(jobID, metrics.FileTypeError), jobErrorStorageName(jobID)))
		if len(errLines) != 1 || errLines[0].ID != stableBatchRequestID(jobID, "req-2") {
			t.Fatalf("error artifact = %+v, want the not-executed req-2 line", errLines)
		}
		if errLines[0].Error == nil || errLines[0].Error.Code != string(batch_types.ErrCodeBatchCancelled) {
			t.Fatalf("error line is not batch_cancelled: %+v", errLines[0])
		}

		if got := env.asyncClient.submitCount(); got != 0 {
			t.Fatalf("submitted requests = %d, want 0 (a terminal-bound pickup dispatches nothing)", got)
		}
	})

	t.Run("shutdown guard requeues through the resumable store", func(t *testing.T) {
		const jobID = "job-resumable-poll-shutdown"
		ctx := testLoggerCtx(t)

		env := setupDurableJob(t, jobID, []string{"test-model"})

		// Put the mock batch in the activated state a requeued claim implies.
		// The Spec only needs to parse: runJob is never launched.
		env.jobItem.Spec = mustJSON(t, openai.BatchSpec{InputFileID: "unused"})
		env.jobItem.Status = mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusInProgress})
		if err := env.dbClient.DBStore(ctx, env.jobItem); err != nil {
			t.Fatalf("DBStore job item: %v", err)
		}
		err := env.resumable.ActivateResumableBatch(ctx, &db.ResumableActivation{
			BatchID:     jobID,
			ProcessorID: "test-processor",
			Epoch:       1,
			LeaseTTL:    time.Minute,
			Entries: []db.ResumableActivationEntry{
				{RequestID: "r-1", Ordinal: 0, CustomID: "c-1", Status: string(db.ResumableEntryPending)},
			},
		})
		if err != nil {
			t.Fatalf("ActivateResumableBatch: %v", err)
		}

		// Cancel the polling context right after the claim: the pre-launch
		// guard fires with the task in hand.
		runCtx, cancel := context.WithCancel(ctx)
		pq := &spyPQ{inner: env.pqClient, afterDequeueFn: cancel}
		env.p.poller = NewPoller(pq, env.dbClient)

		if err := pq.PQEnqueue(ctx, &db.BatchJobPriority{
			ID:        jobID,
			SLO:       time.Now().Add(1 * time.Hour),
			Epoch:     1,
			Resumable: true,
		}); err != nil {
			t.Fatalf("PQEnqueue task: %v", err)
		}

		if err := env.p.runPollingLoop(runCtx, runCtx); err != nil {
			t.Fatalf("runPollingLoop: %v", err)
		}

		// The batch is back in the queue via the fenced requeue: ownership
		// released, epoch bumped, one recovery attempt — and no terminal
		// write or legacy re-enqueue.
		owner, epoch, attempts := env.resumable.BatchOwnership(jobID)
		if owner != "" || epoch != 2 || attempts != 1 {
			t.Fatalf("ownership after shutdown requeue = (%q, %d, %d), want released at epoch 2 with 1 attempt", owner, epoch, attempts)
		}
		assertJobStatus(t, env.dbClient, jobID, openai.BatchStatusInProgress)
		if calls := pq.EnqueueCalls(); calls != 1 {
			t.Fatalf("enqueue calls = %d, want only the seed (legacy re-enqueue must be skipped)", calls)
		}
	})
}

// TestRunJob_Resumable_PickupRouting covers the terminal-bound pickup matrix
// of a requeued resumable task: every branch finalizes with partial artifacts
// (non-terminal entries become not-executed error lines) in its terminal
// status, without dispatching anything.
func TestRunJob_Resumable_PickupRouting(t *testing.T) {
	tests := []struct {
		name           string
		stage          func(env *durableJobEnv, task *db.BatchJobPriority)
		want           openai.BatchStatus
		wantNotExecVer batch_types.BatchErrorCode
	}{
		{
			name: "SLO-expired pickup finalizes as expired",
			stage: func(env *durableJobEnv, task *db.BatchJobPriority) {
				task.SLO = time.Now().Add(-1 * time.Second)
			},
			want:           openai.BatchStatusExpired,
			wantNotExecVer: batch_types.ErrCodeBatchExpired,
		},
		{
			name: "exhausted recovery budget finalizes as failed",
			stage: func(env *durableJobEnv, task *db.BatchJobPriority) {
				env.jobItem.RecoveryAttempts = maxRecoveryAttempts + 1
			},
			want:           openai.BatchStatusFailed,
			wantNotExecVer: batch_types.ErrCodeBatchFailed,
		},
	}
	for i, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jobID := fmt.Sprintf("job-resumable-route-%d", i)

			env := setupDurableJob(t, jobID, []string{"test-model", "test-model"})
			env.jobItem.Spec = mustJSON(t, openai.BatchSpec{
				InputFileID: "file-input-" + jobID,
				Endpoint:    openai.EndpointChatCompletions,
			})
			env.jobItem.Status = mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusInProgress})
			env.jobItem.Epoch = 2

			task := &db.BatchJobPriority{
				ID:        jobID,
				SLO:       time.Now().Add(1 * time.Hour),
				Epoch:     2,
				Resumable: true,
			}
			tt.stage(env, task)

			if err := env.dbClient.DBStore(testLoggerCtx(t), env.jobItem); err != nil {
				t.Fatalf("DBStore job item: %v", err)
			}
			env.stageResumedClaim(t, 2,
				checkpointedLineEntry(t, env, jobID, "req-1", 0),
				pendingLineEntry(jobID, "req-2", 1),
			)

			env.runResumedJob(t, task)

			status := durableJobStatus(t, env)
			if status.Status != tt.want {
				t.Fatalf("status = %s, want %s", status.Status, tt.want)
			}
			wantCounts := openai.BatchRequestCounts{Total: 2, Completed: 1, Failed: 1}
			if status.RequestCounts != wantCounts {
				t.Fatalf("counts = %+v, want %+v", status.RequestCounts, wantCounts)
			}

			outLines := readArtifactLines(t, env,
				ucom.FileStorageName(stableFileID(jobID, metrics.FileTypeOutput), jobOutputStorageName(jobID)))
			if len(outLines) != 1 || outLines[0].ID != stableBatchRequestID(jobID, "req-1") {
				t.Fatalf("output artifact = %+v, want the checkpointed req-1 line", outLines)
			}

			errLines := readArtifactLines(t, env,
				ucom.FileStorageName(stableFileID(jobID, metrics.FileTypeError), jobErrorStorageName(jobID)))
			if len(errLines) != 1 || errLines[0].ID != stableBatchRequestID(jobID, "req-2") {
				t.Fatalf("error artifact = %+v, want the not-executed req-2 line", errLines)
			}
			if errLines[0].Error == nil || errLines[0].Error.Code != string(tt.wantNotExecVer) {
				t.Fatalf("error line code = %+v, want %s", errLines[0].Error, tt.wantNotExecVer)
			}

			if got := env.asyncClient.submitCount(); got != 0 {
				t.Fatalf("submitted requests = %d, want 0 (a terminal-bound pickup dispatches nothing)", got)
			}
		})
	}
}

// TestRunJob_Resumable_ResumeSkipsCheckpointedEntries verifies the resumed
// incarnation of a requeued batch end to end: the persisted entry set passes
// the manifest integrity check, checkpointed entries are skipped at the source
// (never re-dispatched), and finalization streams both incarnations' result
// objects into one artifact in input order.
func TestRunJob_Resumable_ResumeSkipsCheckpointedEntries(t *testing.T) {
	const jobID = "job-resumable-resume"

	env := setupDurableJob(t, jobID, []string{"test-model", "test-model"})

	// The previous incarnation checkpointed req-1 before dying; the requeued
	// row was re-claimed at epoch 2 (the requeue/claim bookkeeping is
	// PG-tested — the mock stages the claimed state directly).
	env.jobItem.Status = mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusInProgress})
	env.jobItem.Epoch = 2
	if err := env.dbClient.DBStore(testLoggerCtx(t), env.jobItem); err != nil {
		t.Fatalf("DBStore job item: %v", err)
	}
	env.stageResumedClaim(t, 2,
		checkpointedLineEntry(t, env, jobID, "req-1", 0),
		pendingLineEntry(jobID, "req-2", 1),
	)

	done := env.runResumedJobAsync(t, &db.BatchJobPriority{
		ID:        jobID,
		SLO:       time.Now().Add(1 * time.Hour),
		Epoch:     2,
		Resumable: true,
	})

	// Answer every submitted request — the resumed run must submit only the
	// unfinished entry — until the run returns.
	go func() {
		for {
			select {
			case req := <-env.asyncClient.submittedCh:
				env.asyncClient.deliver(req.RequestID, map[string]any{"choices": []any{map[string]any{"text": "ok"}}})
			case <-done:
				return
			}
		}
	}()
	<-done

	status := durableJobStatus(t, env)
	if status.Status != openai.BatchStatusCompleted {
		t.Fatalf("status = %s, want completed", status.Status)
	}
	wantCounts := openai.BatchRequestCounts{Total: 2, Completed: 2, Failed: 0}
	if status.RequestCounts != wantCounts {
		t.Fatalf("counts = %+v, want %+v", status.RequestCounts, wantCounts)
	}

	// Only the unfinished entry was re-dispatched.
	if got := env.asyncClient.submitCount(); got != 1 {
		t.Fatalf("submitted requests = %d, want 1", got)
	}

	// Output artifact: both incarnations' objects, in input order.
	outLines := readArtifactLines(t, env,
		ucom.FileStorageName(stableFileID(jobID, metrics.FileTypeOutput), jobOutputStorageName(jobID)))
	if len(outLines) != 2 {
		t.Fatalf("output artifact lines = %d, want 2", len(outLines))
	}
	wantIDs := []string{stableBatchRequestID(jobID, "req-1"), stableBatchRequestID(jobID, "req-2")}
	for i, want := range wantIDs {
		if outLines[i].ID != want {
			t.Fatalf("output line %d id = %s, want %s", i, outLines[i].ID, want)
		}
	}
	// Line 0 carries the object the earlier incarnation checkpointed; line 1
	// this run's result.
	for i, wantText := range []string{"earlier-incarnation", "ok"} {
		choices, ok := outLines[i].Response.Body["choices"].([]any)
		if !ok || len(choices) != 1 {
			t.Fatalf("output line %d body = %+v, want one choice", i, outLines[i].Response.Body)
		}
		if text, ok := choices[0].(map[string]any)["text"]; !ok || text != wantText {
			t.Fatalf("output line %d choice text = %v, want %q", i, text, wantText)
		}
	}
}

// TestRunJob_Resumable_IntegrityMismatchFailsTheJob verifies that a persisted
// entry set that no longer matches the manifest re-derived from the input file
// fails the job instead of resuming a diverged batch.
func TestRunJob_Resumable_IntegrityMismatchFailsTheJob(t *testing.T) {
	const jobID = "job-resumable-integrity"

	env := setupDurableJob(t, jobID, []string{"test-model", "test-model"})

	// The input has two lines but only one entry survived: allowlist drift
	// between incarnations must be caught, not resumed.
	env.jobItem.Status = mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusInProgress})
	env.jobItem.Epoch = 2
	if err := env.dbClient.DBStore(testLoggerCtx(t), env.jobItem); err != nil {
		t.Fatalf("DBStore job item: %v", err)
	}
	env.stageResumedClaim(t, 2, checkpointedLineEntry(t, env, jobID, "req-1", 0))

	env.runResumedJob(t, &db.BatchJobPriority{
		ID:        jobID,
		SLO:       time.Now().Add(1 * time.Hour),
		Epoch:     2,
		Resumable: true,
	})

	assertJobStatus(t, env.dbClient, jobID, openai.BatchStatusFailed)
	if got := env.asyncClient.submitCount(); got != 0 {
		t.Fatalf("submitted requests = %d, want 0 (a diverged batch must not resume)", got)
	}
}

// TestRunJob_Resumable_MidRunOwnershipLoss verifies the zombie-abort contract
// when ownership is taken away mid-run: the fenced checkpoint surfaces
// db.ErrConflict, runJob aborts via handleOwnershipLost, and no terminal write
// or entry mutation happens — the new owner owns the batch's terminal state.
func TestRunJob_Resumable_MidRunOwnershipLoss(t *testing.T) {
	const jobID = "job-resumable-midrun-fence"

	env := setupDurableJob(t, jobID, []string{"test-model", "test-model"})

	env.jobItem.Status = mustJSON(t, openai.BatchStatusInfo{Status: openai.BatchStatusInProgress})
	env.jobItem.Epoch = 2
	if err := env.dbClient.DBStore(testLoggerCtx(t), env.jobItem); err != nil {
		t.Fatalf("DBStore job item: %v", err)
	}
	env.stageResumedClaim(t, 2,
		pendingLineEntry(jobID, "req-1", 0),
		pendingLineEntry(jobID, "req-2", 1),
	)

	done := env.runResumedJobAsync(t, &db.BatchJobPriority{
		ID:        jobID,
		SLO:       time.Now().Add(1 * time.Hour),
		Epoch:     2,
		Resumable: true,
	})

	// Let the run dispatch both entries, then take ownership away the way the
	// GC reconciler does, and answer the results: their checkpoints must
	// fence out and abort the run.
	var submitted []string
	for range 2 {
		req := <-env.asyncClient.submittedCh
		submitted = append(submitted, req.RequestID)
	}
	if err := env.resumable.RequeueResumableOrphan(context.Background(), jobID, env.jobItem.ProcessorID, 2, false); err != nil {
		t.Fatalf("RequeueResumableOrphan: %v", err)
	}
	for _, id := range submitted {
		env.asyncClient.deliver(id, map[string]any{"choices": []any{map[string]any{"text": "late"}}})
	}

	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("runJob did not return after the fenced checkpoint")
	}

	// No terminal write: the row stays in_progress for the new owner.
	assertJobStatus(t, env.dbClient, jobID, openai.BatchStatusInProgress)

	// Ownership was released to the requeue state; nothing was checkpointed
	// after the takeover.
	owner, epoch, attempts := env.resumable.BatchOwnership(jobID)
	if owner != "" || epoch != 3 || attempts != 1 {
		t.Fatalf("ownership = (%q, %d, %d), want released at epoch 3 with 1 attempt", owner, epoch, attempts)
	}
	state, err := env.resumable.EntryResumeState(context.Background(), jobID)
	if err != nil {
		t.Fatalf("EntryResumeState: %v", err)
	}
	if state.Completed != 0 || state.Failed != 0 || state.InFlight != 2 {
		t.Fatalf("entry state = %+v, want 2 in-flight and nothing checkpointed", state)
	}
}

func TestRenewBatchLease(t *testing.T) {
	const jobID = "job-lease"

	seedActivated := func(t *testing.T) *mockdb.MockResumableStore {
		t.Helper()
		store := mockdb.NewMockResumableStore()
		store.SeedBatch(jobID, "test-processor", 1)
		err := store.ActivateResumableBatch(context.Background(), &db.ResumableActivation{
			BatchID:     jobID,
			ProcessorID: "test-processor",
			Epoch:       1,
			LeaseTTL:    time.Minute,
			Entries: []db.ResumableActivationEntry{
				{RequestID: "r-1", Ordinal: 0, CustomID: "c-1", Status: string(db.ResumableEntryPending)},
			},
		})
		if err != nil {
			t.Fatalf("ActivateResumableBatch: %v", err)
		}
		return store
	}

	newLeaseProcessor := func(t *testing.T, store *mockdb.MockResumableStore) *Processor {
		t.Helper()
		cfg := config.NewConfig()
		cfg.Resumable.LeaseDuration = 10 * time.Millisecond
		return mustNewProcessor(t, cfg, &clientset.Clientset{Resumable: store})
	}

	jobItem := &db.BatchItem{
		BaseIndexes: db.BaseIndexes{ID: jobID},
		Epoch:       1,
	}

	t.Run("fenced renewal trips onOwnershipLost", func(t *testing.T) {
		store := seedActivated(t)
		store.MarkTerminal(jobID) // a terminal transition fences the owner out

		p := newLeaseProcessor(t, store)
		lost := make(chan struct{})
		go p.renewBatchLease(context.Background(), jobItem, func() { close(lost) }, make(chan struct{}))

		select {
		case <-lost:
		case <-time.After(5 * time.Second):
			t.Fatal("fenced renewal did not trip onOwnershipLost")
		}
	})

	t.Run("successful renewals continue until done closes", func(t *testing.T) {
		store := seedActivated(t)
		// Freeze the store's clock: the 10ms lease leaves a wall-clock-thin
		// renewal window, and under load a scheduler hiccup lets it lapse and
		// fences the owner out. Renewals succeeding — not expiry — is what
		// this subtest exercises.
		clock := time.Now()
		store.Now = func() time.Time { return clock }
		p := newLeaseProcessor(t, store)

		lost := make(chan struct{})
		exited := make(chan struct{})
		done := make(chan struct{})
		go func() {
			defer close(exited)
			p.renewBatchLease(context.Background(), jobItem, func() { close(lost) }, done)
		}()

		// Several renewal ticks pass without an ownership loss…
		time.Sleep(50 * time.Millisecond)
		select {
		case <-lost:
			t.Fatal("successful renewals must not trip onOwnershipLost")
		default:
		}

		close(done)
		select {
		case <-exited:
		case <-time.After(5 * time.Second):
			t.Fatal("renewBatchLease did not stop after done closed")
		}
	})
}
