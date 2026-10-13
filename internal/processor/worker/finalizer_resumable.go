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
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/go-logr/logr"
	"github.com/google/uuid"

	db "github.com/llm-d/llm-d-batch-gateway/internal/database/api"
	filesapi "github.com/llm-d/llm-d-batch-gateway/internal/files_store/api"
	"github.com/llm-d/llm-d-batch-gateway/internal/processor/batchctx"
	"github.com/llm-d/llm-d-batch-gateway/internal/processor/metrics"
	"github.com/llm-d/llm-d-batch-gateway/internal/shared/openai"
	batch_types "github.com/llm-d/llm-d-batch-gateway/internal/shared/types"
	ucom "github.com/llm-d/llm-d-batch-gateway/internal/util/com"
	"github.com/llm-d/llm-d-batch-gateway/internal/util/logging"
	uotel "github.com/llm-d/llm-d-batch-gateway/internal/util/otel"
)

// resumableEntriesPageSize bounds one ListBatchEntries page during artifact
// assembly (keyset pagination by ordinal).
const resumableEntriesPageSize = 500

// finalizeResumableJob finalizes a durable batch: it streams every entry's
// checkpointed result object into the output and error artifacts and
// transitions the batch to completed. Both the finalizing and terminal
// writes are fenced on the persisted resumable flag, so only the incarnation
// that still owns the activated batch may terminalize it. Retry-safe
// finalization of interrupted batches lands with the hardened finalizer.
func (p *Processor) finalizeResumableJob(
	ctx context.Context,
	updater *StatusUpdater,
	dbJob *db.BatchItem,
	jobInfo *batch_types.JobInfo,
) error {
	logger := logr.FromContextOrDiscard(ctx)
	logger.V(logging.INFO).Info("Starting finalization: resumable job")

	// Detached so a concurrent SIGTERM cannot abort artifact assembly or the
	// final status write; bounded by the configured finalization timeout.
	ioCtx, ioSpan := uotel.DetachedContext(ctx, "finalize-resumable")
	ioCtx, ioCancel := context.WithTimeout(ioCtx, p.cfg.Resumable.FinalizeTimeout)
	defer ioCancel()
	defer ioSpan.End()

	resumable := true
	dbJob.ExpectedResumable = &resumable

	// in_progress → finalizing, before artifact assembly so the API server
	// can reject cancel requests once finalization has begun.
	if err := updater.UpdatePersistentStatus(ioCtx, dbJob, openai.BatchStatusFinalizing, nil, nil); err != nil {
		return fmt.Errorf("failed to update job status to finalizing: %w", err)
	}

	counts, outputFileID, errorFileID, err := p.assembleResumableArtifacts(ioCtx, dbJob, jobInfo, "")
	if err != nil {
		logger.Error(err, "Failed to assemble resumable artifacts")
		if failErr := updater.UpdateFailedStatus(ioCtx, dbJob, counts, outputFileID, errorFileID); failErr != nil {
			return fmt.Errorf("artifact assembly failed (%w) and fallback status write also failed: %w", err, failErr)
		}
		return fmt.Errorf("assemble resumable artifacts: %w", errFinalizeFailedOver)
	}

	// Best-effort: honour a user cancel that arrived during finalization —
	// same contract as the legacy finalizer.
	if errors.Is(context.Cause(ctx), batchctx.ErrCancelled) {
		logger.V(logging.INFO).Info("Cancel requested during finalization; finalizing as cancelled")
		if err := updater.UpdateCancelledStatus(ioCtx, dbJob, counts, outputFileID, errorFileID); err != nil {
			logger.Error(err, "Failed to update cancelled status, falling back to failed with file IDs preserved")
			if failErr := updater.UpdateFailedStatus(ioCtx, dbJob, counts, outputFileID, errorFileID); failErr != nil {
				return fmt.Errorf("failed to update job status to cancelled (%w) and fallback to failed also failed: %w", err, failErr)
			}
			return fmt.Errorf("cancelled status write failed: %w", errFinalizeFailedOver)
		}
		setRequestCountAttrs(ctx, counts)
		return batchctx.ErrCancelled
	}

	// finalizing → completed
	if err := updater.UpdateCompletedStatus(ioCtx, dbJob, counts, outputFileID, errorFileID); err != nil {
		logger.Error(err, "Failed to update job status to completed, falling back to failed with file IDs preserved")
		if failErr := updater.UpdateFailedStatus(ioCtx, dbJob, counts, outputFileID, errorFileID); failErr != nil {
			return fmt.Errorf("failed to update job status to completed (%w) and fallback to failed also failed: %w", err, failErr)
		}
		return fmt.Errorf("completed status write failed: %w", errFinalizeFailedOver)
	}

	setRequestCountAttrs(ctx, counts)
	logger.V(logging.INFO).Info("Resumable finalization completed", "outputFileID", outputFileID, "errorFileID", errorFileID)
	return nil
}

// finalizeResumableAs terminalizes a durable batch that must not resume
// execution — a requeued row that is cancelling, past its SLO, or past its
// recovery budget. It streams the checkpointed artifacts (non-terminal
// entries become not-executed error lines) and writes the given terminal
// status. Cleanup and metrics belong to the caller. A failed terminal write
// returns the error: the row stays non-terminal and the reconciler requeues
// it, so the pickup is retried instead of degraded.
func (p *Processor) finalizeResumableAs(
	ctx context.Context,
	updater *StatusUpdater,
	dbJob *db.BatchItem,
	jobInfo *batch_types.JobInfo,
	status openai.BatchStatus,
	notExecutedCode batch_types.BatchErrorCode,
) error {
	logger := logr.FromContextOrDiscard(ctx)
	logger.V(logging.INFO).Info("Terminalizing resumable job without resuming execution", "status", status)

	ioCtx, ioSpan := uotel.DetachedContext(ctx, "finalize-resumable-as")
	ioCtx, ioCancel := context.WithTimeout(ioCtx, p.cfg.Resumable.FinalizeTimeout)
	defer ioCancel()
	defer ioSpan.End()

	resumable := true
	dbJob.ExpectedResumable = &resumable

	counts, outputFileID, errorFileID, err := p.assembleResumableArtifacts(ioCtx, dbJob, jobInfo, notExecutedCode)
	if err != nil {
		logger.Error(err, "Failed to assemble resumable artifacts")
		if failErr := updater.UpdateFailedStatus(ioCtx, dbJob, counts, outputFileID, errorFileID); failErr != nil {
			return fmt.Errorf("artifact assembly failed (%w) and fallback status write also failed: %w", err, failErr)
		}
		return fmt.Errorf("assemble resumable artifacts: %w", errFinalizeFailedOver)
	}

	var writeErr error
	switch status {
	case openai.BatchStatusCancelled:
		writeErr = updater.UpdateCancelledStatus(ioCtx, dbJob, counts, outputFileID, errorFileID)
	case openai.BatchStatusExpired:
		writeErr = updater.UpdateExpiredStatus(ioCtx, dbJob, counts, outputFileID, errorFileID)
	default:
		writeErr = updater.UpdateFailedStatus(ioCtx, dbJob, counts, outputFileID, errorFileID)
	}
	if writeErr != nil {
		return fmt.Errorf("update job status to %s: %w", status, writeErr)
	}

	setRequestCountAttrs(ctx, counts)
	logger.V(logging.INFO).Info("Resumable job terminalized", "status", status,
		"outputFileID", outputFileID, "errorFileID", errorFileID)
	return nil
}

// assembleResumableArtifacts streams every entry's result object, in input
// order, into the batch's output and error artifacts and returns the final
// request counts and file IDs. Each object's size and SHA-256 are verified
// while streaming. Artifact file IDs are derived from the batch, so a
// re-finalization targets the same keys.
//
// notExecutedCode renders non-terminal entries as synthetic not-executed
// error lines, for terminalizing a batch that must not resume execution; an
// empty code demands every entry be terminal (the completed path).
func (p *Processor) assembleResumableArtifacts(
	ctx context.Context,
	dbJob *db.BatchItem,
	jobInfo *batch_types.JobInfo,
	notExecutedCode batch_types.BatchErrorCode,
) (*openai.BatchRequestCounts, string, string, error) {
	jobID := jobInfo.JobID
	folder, err := ucom.GetFolderNameByTenantID(jobInfo.TenantID)
	if err != nil {
		return nil, "", "", fmt.Errorf("resolve artifact folder: %w", err)
	}

	output := &artifactWriter{
		storage: p.files.storage,
		folder:  folder,
		key:     ucom.FileStorageName(stableFileID(jobID, metrics.FileTypeOutput), jobOutputStorageName(jobID)),
		fileID:  stableFileID(jobID, metrics.FileTypeOutput),
	}
	errorW := &artifactWriter{
		storage: p.files.storage,
		folder:  folder,
		key:     ucom.FileStorageName(stableFileID(jobID, metrics.FileTypeError), jobErrorStorageName(jobID)),
		fileID:  stableFileID(jobID, metrics.FileTypeError),
	}
	// Abort is a no-op for a writer that never opened.
	defer func() {
		_ = output.abort(ctx)
		_ = errorW.abort(ctx)
	}()

	counts := &openai.BatchRequestCounts{}
	afterOrdinal := int64(-1)
	for {
		entries, err := p.resumable.ListBatchEntries(ctx, jobID, afterOrdinal, resumableEntriesPageSize)
		if err != nil {
			return nil, "", "", fmt.Errorf("list batch entries: %w", err)
		}
		if len(entries) == 0 {
			break
		}
		for _, e := range entries {
			afterOrdinal = e.Ordinal
			counts.Total++
			var w *artifactWriter
			switch e.Status {
			case db.ResumableEntryCompleted:
				counts.Completed++
				w = output
			case db.ResumableEntryFailed:
				counts.Failed++
				// A failed entry without an error object preserved a non-2xx
				// HTTP response — an output line, matching the sync
				// collector's routing.
				w = output
				if e.ErrorCode != "" {
					w = errorW
				}
			default:
				if notExecutedCode == "" {
					// Execution cannot succeed with unresolved entries; the
					// checkpointer resolves everything it drains.
					return nil, "", "", fmt.Errorf("entry %s is still %s", e.RequestID, e.Status)
				}
				counts.Failed++
				if err := errorW.writeNotExecuted(ctx, e, notExecutedCode); err != nil {
					return nil, "", "", err
				}
				continue
			}
			if err := w.streamEntry(ctx, e); err != nil {
				return nil, "", "", err
			}
		}
		if len(entries) < resumableEntriesPageSize {
			break
		}
	}

	outputFileID, err := output.commit(ctx, p, dbJob, jobInfo, jobOutputStorageName(jobID))
	if err != nil {
		return nil, "", "", err
	}
	errorFileID, err := errorW.commit(ctx, p, dbJob, jobInfo, jobErrorStorageName(jobID))
	if err != nil {
		return nil, "", "", err
	}
	return counts, outputFileID, errorFileID, nil
}

// artifactWriter lazily opens and finally commits one finalization artifact.
// An artifact with no entries is never created; its file ID stays empty.
type artifactWriter struct {
	storage filesapi.BatchFilesClient
	folder  string
	key     string
	fileID  string
	writer  filesapi.StreamWriter
}

// ensure opens the stream writer on first use.
func (w *artifactWriter) ensure(ctx context.Context) error {
	if w.writer != nil {
		return nil
	}
	writer, err := w.storage.OpenStream(ctx, w.key, w.folder)
	if err != nil {
		return fmt.Errorf("open artifact stream %s: %w", w.key, err)
	}
	w.writer = writer
	return nil
}

// streamEntry copies one entry's result object into the artifact, verifying
// its size and SHA-256 while streaming.
func (w *artifactWriter) streamEntry(ctx context.Context, e *db.ResumableEntry) error {
	if err := w.ensure(ctx); err != nil {
		return err
	}
	reader, _, err := w.storage.Retrieve(ctx, e.ResultKey, w.folder)
	if err != nil {
		return fmt.Errorf("retrieve result %s: %w", e.ResultKey, err)
	}
	defer reader.Close()

	hasher := sha256.New()
	size, err := io.Copy(io.MultiWriter(w.writer, hasher), reader)
	if err != nil {
		return fmt.Errorf("stream result %s: %w", e.ResultKey, err)
	}
	if size != e.ResultBytes || hex.EncodeToString(hasher.Sum(nil)) != e.ResultSHA256 {
		return fmt.Errorf("result %s failed integrity check: got %d bytes, want %d", e.ResultKey, size, e.ResultBytes)
	}
	return nil
}

// writeNotExecuted renders a non-terminal entry as a synthetic not-executed
// error line, so a batch terminalized without resuming still produces a
// complete artifact set.
func (w *artifactWriter) writeNotExecuted(ctx context.Context, e *db.ResumableEntry, code batch_types.BatchErrorCode) error {
	if err := w.ensure(ctx); err != nil {
		return err
	}
	lineBytes, err := json.Marshal(&outputLine{
		ID:       e.RequestID,
		CustomID: e.CustomID,
		Error: &outputError{
			Code:    string(code),
			Message: code.Message(),
		},
	})
	if err != nil {
		return fmt.Errorf("marshal not-executed line for %s: %w", e.RequestID, err)
	}
	lineBytes = append(lineBytes, '\n')
	if _, err := w.writer.Write(lineBytes); err != nil {
		return fmt.Errorf("write not-executed line for %s: %w", e.RequestID, err)
	}
	return nil
}

// commit finalizes the artifact and records its file ID; a writer that never
// opened contributes no file.
func (w *artifactWriter) commit(ctx context.Context, p *Processor, dbJob *db.BatchItem, jobInfo *batch_types.JobInfo, fileName string) (string, error) {
	if w.writer == nil {
		return "", nil
	}
	md, err := w.writer.Commit(ctx)
	if err != nil {
		return "", fmt.Errorf("commit artifact %s: %w", w.key, err)
	}
	w.writer = nil
	if err := p.storeFileRecord(ctx, w.fileID, fileName, jobInfo.TenantID, md.Size, dbJob.Tags); err != nil {
		return "", err
	}
	return w.fileID, nil
}

// abort discards the write; a no-op after a successful commit.
func (w *artifactWriter) abort(ctx context.Context) error {
	if w.writer == nil {
		return nil
	}
	err := w.writer.Abort(ctx)
	w.writer = nil
	return err
}

// stableFileID derives an artifact file ID from the batch and file type, so
// re-finalization writes the same artifact keys.
func stableFileID(batchID string, fileType metrics.FileType) string {
	u := uuid.NewSHA1(uuid.NameSpaceOID, []byte(batchID+"\x00"+string(fileType)))
	return "file_" + u.String()
}
