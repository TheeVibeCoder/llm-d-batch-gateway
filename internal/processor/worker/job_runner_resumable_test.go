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
	asyncClient *fakeAsyncInferenceClient
	jobItem     *db.BatchItem
	jobInfo     *batch_types.JobInfo
}

// durableInputLine builds one input line for the given model and custom ID.
func durableInputLine(model, customID string) string {
	return fmt.Sprintf(`{"custom_id":%q,"method":"POST","url":"/v1/chat/completions","body":{"model":%q,"messages":[{"role":"user","content":"hi"}]}}`+"\n", customID, model)
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
	if err := fileDBClient.DBStore(ctx, fileItem); err != nil {
		t.Fatalf("DBStore file item: %v", err)
	}

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
