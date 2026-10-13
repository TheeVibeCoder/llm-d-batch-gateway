package pipeline

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-batch-gateway/internal/processor/metrics"
	batch_types "github.com/llm-d/llm-d-batch-gateway/internal/shared/types"
)

// outputLine is the JSONL format for one result, matching the OpenAI batch output schema.
type outputLine struct {
	ID       string                    `json:"id"`
	CustomID string                    `json:"custom_id"`
	Response *batch_types.ResponseData `json:"response"`
	Error    *OutputError              `json:"error"`
}

const fileBufferSize = 1024 * 1024

// Succeeded reports whether a result is a success line: no error object and
// an HTTP 200 response. It matches the outputLine classification.
func (r *ResultItem) Succeeded() bool {
	return r.Error == nil && r.Response != nil && r.Response.StatusCode == 200
}

// ErrorCode classifies a non-success result for error reporting: the output
// error's code when present, otherwise the HTTP status.
func (r *ResultItem) ErrorCode() string {
	if r.Error != nil {
		return r.Error.Code
	}
	if r.Response != nil {
		return fmt.Sprintf("http_%d", r.Response.StatusCode)
	}
	return "unknown"
}

// MarshalResultLine renders one result as an output JSONL line (with its
// trailing newline) and returns the line bytes, its SHA-256 hex digest, and
// its size in bytes — the metadata the durable pipeline checkpoints per entry.
func MarshalResultLine(msg *ResultItem) ([]byte, string, int64, error) {
	line := &outputLine{
		ID:       msg.RequestID,
		CustomID: msg.CustomID,
		Response: msg.Response,
		Error:    msg.Error,
	}
	lineBytes, err := json.Marshal(line)
	if err != nil {
		return nil, "", 0, fmt.Errorf("marshal output for %s: %w", msg.RequestID, err)
	}
	lineBytes = append(lineBytes, '\n')
	sum := sha256.Sum256(lineBytes)
	return lineBytes, hex.EncodeToString(sum[:]), int64(len(lineBytes)), nil
}

// ResultCollector writes ResultItem values to JSONL and records progress.
// Terminal actor — no out channel.
type ResultCollector struct {
	output               *bufio.Writer
	errors               *bufio.Writer
	pending              *PendingRequests
	tracker              *ProgressTracker
	logger               logr.Logger
	onPersistenceFailure func()
}

func NewResultCollector(outputFile, errorFile *os.File, pending *PendingRequests, tracker *ProgressTracker, logger logr.Logger) *ResultCollector {
	output := bufio.NewWriterSize(outputFile, fileBufferSize)
	errors := bufio.NewWriterSize(errorFile, fileBufferSize)

	return &ResultCollector{output: output, errors: errors, pending: pending, tracker: tracker, logger: logger}
}

// Drain reads results until resultCh is closed, then flushes.
// Both dispatchers close the channel when done, so this always terminates.
// On a write error, Drain continues reading but skips further writes. This
// is intentional: the dispatcher sends results to resultCh and blocks if
// nobody reads; returning early would deadlock the pipeline. The error
// is returned after the channel closes, causing executeJobAsync to route
// the job to handleFailed (which uploads whatever partial output is on disk).
func (c *ResultCollector) Drain(ctx context.Context, resultCh <-chan ResultItem) error {
	var firstErr error
	for msg := range resultCh {
		if !c.pending.Resolve(&msg) {
			continue
		}
		if !msg.SubmittedAt.IsZero() {
			metrics.DecProcessorInflightRequests()
			metrics.DecModelInflightRequests(msg.ModelID)
			metrics.RecordModelRequestExecutionDuration(time.Since(msg.SubmittedAt), msg.ModelID)
		}
		if firstErr != nil {
			continue
		}
		if err := c.Receive(msg); err != nil {
			firstErr = err
			c.logger.Error(err, "Persistence failure, skipping further writes")
			if c.onPersistenceFailure != nil {
				c.onPersistenceFailure()
			}
		}
	}
	if flushErr := c.flushFiles(); flushErr != nil {
		return flushErr
	}
	if firstErr != nil {
		return firstErr
	}
	if ctx.Err() != nil {
		c.logger.Info("Context cancelled after drain completed", "err", ctx.Err())
	}
	return nil
}

func (c *ResultCollector) Receive(msg ResultItem) error {
	lineBytes, _, _, err := MarshalResultLine(&msg)
	if err != nil {
		return err
	}

	w := c.output
	if msg.Error != nil {
		w = c.errors
	}
	if _, err := w.Write(lineBytes); err != nil {
		return fmt.Errorf("write output for %s: %w", msg.RequestID, err)
	}

	if msg.Succeeded() {
		c.tracker.RecordSuccess(msg)
		if msg.Response != nil {
			recordTokenUsage(msg.Response.Body, msg.ModelID, c.logger)
		}
	} else {
		c.tracker.RecordFailure(fmt.Errorf("%s: %s", msg.RequestID, msg.ErrorCode()))
		metrics.RecordRequestError(msg.ModelID)
	}

	return nil
}

// setPersistenceFailureHook arms the executor's persistence-failure abort.
func (c *ResultCollector) setPersistenceFailureHook(fn func()) {
	c.onPersistenceFailure = fn
}

func recordTokenUsage(body map[string]any, model string, logger logr.Logger) {
	usage, ok := body["usage"].(map[string]any)
	if !ok {
		return
	}
	prompt, pOK := toFloat64(usage["prompt_tokens"])
	completion, cOK := toFloat64(usage["completion_tokens"])
	if !pOK && !cOK {
		return
	}
	if prompt < 0 || completion < 0 {
		logger.Info("Negative token values, skipping metrics",
			"prompt_tokens", prompt, "completion_tokens", completion)
		return
	}
	metrics.RecordTokenUsage(prompt, completion, model)
}

func toFloat64(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

func (c *ResultCollector) flushFiles() (err error) {
	if err = c.output.Flush(); err != nil {
		c.logger.Error(err, "Failed to flush output file")
	}
	if errErr := c.errors.Flush(); errErr != nil {
		c.logger.Error(errErr, "Failed to flush error file")
		if err == nil {
			err = errErr
		}
	}
	return
}
