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

package postgresql

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/llm-d/llm-d-batch-gateway/internal/database/api"
)

// resumablePgEnv bundles the real clients and pool the pg-backed tests use.
type resumablePgEnv struct {
	ctx     context.Context
	store   *PostgresResumableStore
	batchDB *PostgresBatchDBClient
	queue   *PostgresBatchQueueClient
	pool    *pgxpool.Pool
}

// newResumablePgEnv prepares a clean database and the clients under test.
func newResumablePgEnv(t *testing.T) *resumablePgEnv {
	t.Helper()
	url := os.Getenv("TEST_POSTGRES_URL")
	if url == "" {
		t.Skip("TEST_POSTGRES_URL not set")
	}
	ctx := context.Background()

	cfg := &PostgreSQLConfig{Url: url}
	env := &resumablePgEnv{ctx: ctx}
	var err error
	if env.store, err = NewPostgresResumableStore(ctx, cfg); err != nil {
		t.Fatalf("NewPostgresResumableStore: %v", err)
	}
	if env.batchDB, err = NewPostgresBatchDBClient(ctx, cfg); err != nil {
		t.Fatalf("NewPostgresBatchDBClient: %v", err)
	}
	if env.queue, err = NewPostgresBatchQueueClient(ctx, cfg, testProcessorID); err != nil {
		t.Fatalf("NewPostgresBatchQueueClient: %v", err)
	}
	if env.pool, err = pgxpool.New(ctx, url); err != nil {
		t.Fatalf("pgxpool: %v", err)
	}
	t.Cleanup(func() {
		env.pool.Close()
		_ = env.queue.Close()
		_ = env.batchDB.Close()
		_ = env.store.Close()
	})

	// CASCADE: batch_entries holds a foreign key to batch_items.
	if _, err := env.pool.Exec(ctx, "TRUNCATE batch_items CASCADE"); err != nil {
		t.Fatalf("truncate: %v", err)
	}
	return env
}

// seedValidatingBatch stores a queued (validating, unclaimed) batch row.
func (env *resumablePgEnv) seedValidatingBatch(t *testing.T, id string) {
	t.Helper()
	if err := env.batchDB.DBStore(env.ctx, &api.BatchItem{
		BaseIndexes: api.BaseIndexes{ID: id, TenantID: "tenant-1"},
		BaseContents: api.BaseContents{
			Status: []byte(`{"status":"validating"}`),
			Spec:   []byte(`{}`),
		},
		Priority: time.Now().Add(24 * time.Hour).UnixMicro(),
	}); err != nil {
		t.Fatalf("seed batch %s: %v", id, err)
	}
}

// claimFreshBatch seeds a validating batch and claims it through the queue,
// returning the post-dequeue epoch the activation must be fenced on.
func (env *resumablePgEnv) claimFreshBatch(t *testing.T, id string) int64 {
	t.Helper()
	env.seedValidatingBatch(t, id)

	jobs, err := env.queue.PQDequeue(env.ctx, 0, 10)
	if err != nil {
		t.Fatalf("PQDequeue: %v", err)
	}
	for _, job := range jobs {
		if job.ID == id {
			return job.Epoch
		}
	}
	t.Fatalf("dequeue did not claim %s (got %+v)", id, jobs)
	return 0
}

func testActivation(batchID string, epoch int64, entries ...api.ResumableActivationEntry) *api.ResumableActivation {
	return &api.ResumableActivation{
		BatchID:     batchID,
		ProcessorID: testProcessorID,
		Epoch:       epoch,
		LeaseTTL:    5 * time.Minute,
		Entries:     entries,
	}
}

func activationEntries() []api.ResumableActivationEntry {
	return []api.ResumableActivationEntry{
		{RequestID: "batch_req_a", Ordinal: 0, CustomID: "a", Status: "pending"},
		{RequestID: "batch_req_b", Ordinal: 1, CustomID: "b", Status: "pending"},
		{RequestID: "batch_req_r", Ordinal: 2, CustomID: "r", Status: "failed",
			ErrorCode: "model_not_found", ResultKey: "results/j1/batch_req_r.jsonl",
			ResultBytes: 42, ResultSHA256: "sha-r"},
	}
}

func TestActivateResumableBatchPG(t *testing.T) {
	env := newResumablePgEnv(t)
	const batchID = "batch-activate"

	t.Run("inserts entries and flips ownership atomically", func(t *testing.T) {
		epoch := env.claimFreshBatch(t, batchID)

		if err := env.store.ActivateResumableBatch(env.ctx, testActivation(batchID, epoch, activationEntries()...)); err != nil {
			t.Fatalf("ActivateResumableBatch: %v", err)
		}

		var status string
		var resumable bool
		var processor *string
		var leaseExpiry *time.Time
		if err := env.pool.QueryRow(env.ctx,
			`SELECT status->>'status', resumable, processor_id, owner_lease_expires_at
			 FROM batch_items WHERE id = $1`, batchID,
		).Scan(&status, &resumable, &processor, &leaseExpiry); err != nil {
			t.Fatalf("read batch: %v", err)
		}
		if status != "in_progress" || !resumable || processor == nil || *processor != testProcessorID {
			t.Fatalf("batch not activated: status=%s resumable=%v processor=%v", status, resumable, processor)
		}
		if leaseExpiry == nil || !leaseExpiry.After(time.Now()) {
			t.Fatalf("lease not set in the future: %v", leaseExpiry)
		}

		entries, err := env.store.ListBatchEntries(env.ctx, batchID, -1, 10)
		if err != nil {
			t.Fatalf("ListBatchEntries: %v", err)
		}
		if len(entries) != 3 {
			t.Fatalf("expected 3 entries, got %d", len(entries))
		}
		rejected := entries[2]
		if rejected.Status != api.ResumableEntryFailed || rejected.ErrorCode != "model_not_found" ||
			rejected.ResultKey != "results/j1/batch_req_r.jsonl" || rejected.ResultBytes != 42 ||
			rejected.ResultSHA256 != "sha-r" || rejected.Attempts != 1 {
			t.Fatalf("rejected entry not persisted with its error object: %+v", rejected)
		}
	})

	t.Run("conflicts on a stale epoch", func(t *testing.T) {
		epoch := env.claimFreshBatch(t, "batch-stale-epoch")
		err := env.store.ActivateResumableBatch(env.ctx, testActivation("batch-stale-epoch", epoch+1, activationEntries()...))
		if !errors.Is(err, api.ErrConflict) {
			t.Fatalf("expected ErrConflict, got %v", err)
		}
	})

	t.Run("conflicts when another processor owns the row", func(t *testing.T) {
		epoch := env.claimFreshBatch(t, "batch-other-owner")
		act := testActivation("batch-other-owner", epoch, activationEntries()...)
		act.ProcessorID = "processor-other"
		err := env.store.ActivateResumableBatch(env.ctx, act)
		if !errors.Is(err, api.ErrConflict) {
			t.Fatalf("expected ErrConflict, got %v", err)
		}
	})

	t.Run("conflicts when already activated and leaves entries untouched", func(t *testing.T) {
		const id = "batch-double-activate"
		epoch := env.claimFreshBatch(t, id)
		if err := env.store.ActivateResumableBatch(env.ctx, testActivation(id, epoch, activationEntries()...)); err != nil {
			t.Fatalf("first activation: %v", err)
		}
		err := env.store.ActivateResumableBatch(env.ctx, testActivation(id, epoch, activationEntries()...))
		if !errors.Is(err, api.ErrConflict) {
			t.Fatalf("expected ErrConflict on re-activation, got %v", err)
		}
		entries, err := env.store.ListBatchEntries(env.ctx, id, -1, 10)
		if err != nil || len(entries) != 3 {
			t.Fatalf("re-activation must not touch entries: %v %+v", err, entries)
		}
	})

	t.Run("rolls back everything on a malformed entry set", func(t *testing.T) {
		const id = "batch-atomic"
		epoch := env.claimFreshBatch(t, id)
		// Duplicate request IDs violate the entries primary key; the whole
		// statement must fail without flipping the batch.
		bad := []api.ResumableActivationEntry{
			{RequestID: "batch_req_a", Ordinal: 0, CustomID: "a", Status: "pending"},
			{RequestID: "batch_req_a", Ordinal: 1, CustomID: "b", Status: "pending"},
		}
		if err := env.store.ActivateResumableBatch(env.ctx, testActivation(id, epoch, bad...)); err == nil {
			t.Fatal("expected error for duplicate request IDs")
		}

		var status string
		var resumable bool
		if err := env.pool.QueryRow(env.ctx,
			`SELECT status->>'status', resumable FROM batch_items WHERE id = $1`, id,
		).Scan(&status, &resumable); err != nil {
			t.Fatalf("read batch: %v", err)
		}
		if status != "validating" || resumable {
			t.Fatalf("failed activation must not flip the batch: status=%s resumable=%v", status, resumable)
		}
		entries, err := env.store.ListBatchEntries(env.ctx, id, -1, 10)
		if err != nil || len(entries) != 0 {
			t.Fatalf("failed activation must not insert entries: %v %+v", err, entries)
		}
	})
}

func (env *resumablePgEnv) activate(t *testing.T, id string) int64 {
	t.Helper()
	epoch := env.claimFreshBatch(t, id)
	if err := env.store.ActivateResumableBatch(env.ctx, testActivation(id, epoch, activationEntries()...)); err != nil {
		t.Fatalf("ActivateResumableBatch: %v", err)
	}
	return epoch
}

func (env *resumablePgEnv) expireLease(t *testing.T, batchID string) {
	t.Helper()
	if _, err := env.pool.Exec(env.ctx,
		`UPDATE batch_items SET owner_lease_expires_at = NOW() - interval '1 second' WHERE id = $1`,
		batchID,
	); err != nil {
		t.Fatalf("expire lease: %v", err)
	}
}

func TestResumableLeasePG(t *testing.T) {
	env := newResumablePgEnv(t)

	t.Run("renew extends the lease", func(t *testing.T) {
		const id = "batch-renew"
		epoch := env.activate(t, id)

		if err := env.store.RenewResumableBatchLease(env.ctx, id, testProcessorID, epoch, time.Minute); err != nil {
			t.Fatalf("RenewResumableBatchLease: %v", err)
		}
		var expiry time.Time
		if err := env.pool.QueryRow(env.ctx,
			`SELECT owner_lease_expires_at FROM batch_items WHERE id = $1`, id,
		).Scan(&expiry); err != nil {
			t.Fatalf("read lease: %v", err)
		}
		if !expiry.After(time.Now().Add(30 * time.Second)) {
			t.Fatalf("lease not extended: %v", expiry)
		}
	})

	t.Run("renew conflicts on stale epoch or foreign owner", func(t *testing.T) {
		const id = "batch-renew-fence"
		epoch := env.activate(t, id)

		if err := env.store.RenewResumableBatchLease(env.ctx, id, testProcessorID, epoch+1, time.Minute); !errors.Is(err, api.ErrConflict) {
			t.Fatalf("stale epoch: expected ErrConflict, got %v", err)
		}
		if err := env.store.RenewResumableBatchLease(env.ctx, id, "instance-other", epoch, time.Minute); !errors.Is(err, api.ErrConflict) {
			t.Fatalf("foreign owner: expected ErrConflict, got %v", err)
		}
	})

	t.Run("renew conflicts on a lapsed lease", func(t *testing.T) {
		const id = "batch-renew-lapsed"
		epoch := env.activate(t, id)
		env.expireLease(t, id)

		if err := env.store.RenewResumableBatchLease(env.ctx, id, testProcessorID, epoch, time.Minute); !errors.Is(err, api.ErrConflict) {
			t.Fatalf("lapsed lease: expected ErrConflict, got %v", err)
		}
	})
}

func TestRequeueResumableOrphanPG(t *testing.T) {
	env := newResumablePgEnv(t)

	type rowState struct {
		processor       *string
		epoch, attempts int64
		status          string
	}
	readRow := func(t *testing.T, id string) rowState {
		t.Helper()
		var s rowState
		if err := env.pool.QueryRow(env.ctx,
			`SELECT processor_id, epoch, recovery_attempts, status->>'status'
			 FROM batch_items WHERE id = $1`, id,
		).Scan(&s.processor, &s.epoch, &s.attempts, &s.status); err != nil {
			t.Fatalf("read row: %v", err)
		}
		return s
	}

	t.Run("clears ownership, bumps epoch and attempts, keeps status and entries", func(t *testing.T) {
		const id = "batch-requeue"
		epoch := env.activate(t, id)

		if err := env.store.RequeueResumableOrphan(env.ctx, id, testProcessorID, epoch, false); err != nil {
			t.Fatalf("RequeueResumableOrphan: %v", err)
		}

		s := readRow(t, id)
		if s.processor != nil {
			t.Fatalf("ownership not cleared: processor=%v", s.processor)
		}
		if s.epoch != epoch+1 {
			t.Fatalf("epoch not bumped: got %d want %d", s.epoch, epoch+1)
		}
		if s.attempts != 1 {
			t.Fatalf("recovery_attempts not bumped: got %d", s.attempts)
		}
		if s.status != "in_progress" {
			t.Fatalf("status must survive the requeue, got %s", s.status)
		}
		entries, err := env.store.ListBatchEntries(env.ctx, id, -1, 10)
		if err != nil || len(entries) != 3 {
			t.Fatalf("entries must survive the requeue: %v %+v", err, entries)
		}
	})

	t.Run("no-ops when the snapshot is stale", func(t *testing.T) {
		const id = "batch-requeue-race"
		epoch := env.activate(t, id)

		if err := env.store.RequeueResumableOrphan(env.ctx, id, testProcessorID, epoch+5, false); err != nil {
			t.Fatalf("stale snapshot must be a no-op success, got %v", err)
		}
		if s := readRow(t, id); s.processor == nil || *s.processor != testProcessorID {
			t.Fatal("stale requeue must not clear ownership")
		}
	})

	t.Run("wedged-owner trigger requires a lapsed lease", func(t *testing.T) {
		const id = "batch-requeue-wedged"
		epoch := env.activate(t, id)

		if err := env.store.RequeueResumableOrphan(env.ctx, id, testProcessorID, epoch, true); err != nil {
			t.Fatalf("RequeueResumableOrphan: %v", err)
		}
		if s := readRow(t, id); s.processor == nil {
			t.Fatal("unexpired lease must not be requeued by the wedged-owner trigger")
		}

		env.expireLease(t, id)
		if err := env.store.RequeueResumableOrphan(env.ctx, id, testProcessorID, epoch, true); err != nil {
			t.Fatalf("RequeueResumableOrphan: %v", err)
		}
		if s := readRow(t, id); s.processor != nil {
			t.Fatal("lapsed lease must be requeued")
		}
	})

	t.Run("ignores non-resumable rows", func(t *testing.T) {
		const id = "batch-requeue-legacy"
		env.seedValidatingBatch(t, id)
		if _, err := env.pool.Exec(env.ctx,
			`UPDATE batch_items SET processor_id = $2 WHERE id = $1`, id, testProcessorID,
		); err != nil {
			t.Fatalf("claim: %v", err)
		}

		if err := env.store.RequeueResumableOrphan(env.ctx, id, testProcessorID, 0, false); err != nil {
			t.Fatalf("RequeueResumableOrphan: %v", err)
		}
		if s := readRow(t, id); s.processor == nil || *s.processor != testProcessorID {
			t.Fatal("non-resumable row must not be touched")
		}
	})
}

func TestResumableEntryWritesPG(t *testing.T) {
	env := newResumablePgEnv(t)

	const batchID = "batch-entries"
	epoch := env.activate(t, batchID)

	t.Run("mark in flight then checkpoint first-wins", func(t *testing.T) {
		submit, err := env.store.MarkEntryInFlight(env.ctx, batchID, "batch_req_a", testProcessorID, epoch)
		if err != nil || !submit {
			t.Fatalf("MarkEntryInFlight: submit=%v err=%v", submit, err)
		}

		committed, err := env.store.CheckpointEntryResult(env.ctx, batchID, "batch_req_a", testProcessorID, epoch,
			&api.ResumableResult{Status: api.ResumableEntryCompleted,
				ResultKey: "results/j1/batch_req_a.jsonl", ResultBytes: 10, ResultSHA256: "sha-a"})
		if err != nil || !committed {
			t.Fatalf("CheckpointEntryResult: committed=%v err=%v", committed, err)
		}

		// A duplicate delivery of the same result must drop without a commit.
		committed, err = env.store.CheckpointEntryResult(env.ctx, batchID, "batch_req_a", testProcessorID, epoch,
			&api.ResumableResult{Status: api.ResumableEntryCompleted,
				ResultKey: "results/j1/batch_req_a.jsonl", ResultBytes: 10, ResultSHA256: "sha-a"})
		if err != nil || committed {
			t.Fatalf("duplicate checkpoint: committed=%v err=%v", committed, err)
		}

		// A duplicate submission of a terminal entry must be dropped.
		submit, err = env.store.MarkEntryInFlight(env.ctx, batchID, "batch_req_a", testProcessorID, epoch)
		if err != nil || submit {
			t.Fatalf("duplicate MarkEntryInFlight: submit=%v err=%v", submit, err)
		}

		entry, err := env.store.FindEntryByRequestID(env.ctx, batchID, "batch_req_a")
		if err != nil || entry == nil {
			t.Fatalf("FindEntryByRequestID: %v %+v", err, entry)
		}
		if entry.Status != api.ResumableEntryCompleted || entry.Attempts != 1 ||
			entry.ResultKey != "results/j1/batch_req_a.jsonl" || entry.ResultBytes != 10 ||
			entry.ResultSHA256 != "sha-a" || entry.ErrorCode != "" {
			t.Fatalf("checkpoint not persisted faithfully: %+v", entry)
		}
	})

	t.Run("zombie epoch is fenced out", func(t *testing.T) {
		if _, err := env.store.MarkEntryInFlight(env.ctx, batchID, "batch_req_b", testProcessorID, epoch+3); !errors.Is(err, api.ErrConflict) {
			t.Fatalf("stale epoch MarkEntryInFlight: expected ErrConflict, got %v", err)
		}
		_, err := env.store.CheckpointEntryResult(env.ctx, batchID, "batch_req_b", testProcessorID, epoch+3,
			&api.ResumableResult{Status: api.ResumableEntryCompleted, ResultKey: "k", ResultBytes: 1, ResultSHA256: "s"})
		if !errors.Is(err, api.ErrConflict) {
			t.Fatalf("stale epoch CheckpointEntryResult: expected ErrConflict, got %v", err)
		}

		entry, err := env.store.FindEntryByRequestID(env.ctx, batchID, "batch_req_b")
		if err != nil || entry == nil || entry.Status != api.ResumableEntryPending {
			t.Fatalf("zombie must not touch the entry: %v %+v", err, entry)
		}
	})

	t.Run("lapsed lease fences entry writes", func(t *testing.T) {
		env.expireLease(t, batchID)
		if _, err := env.store.MarkEntryInFlight(env.ctx, batchID, "batch_req_b", testProcessorID, epoch); !errors.Is(err, api.ErrConflict) {
			t.Fatalf("lapsed lease MarkEntryInFlight: expected ErrConflict, got %v", err)
		}
		_, err := env.store.CheckpointEntryResult(env.ctx, batchID, "batch_req_b", testProcessorID, epoch,
			&api.ResumableResult{Status: api.ResumableEntryFailed, ErrorCode: "err"})
		if !errors.Is(err, api.ErrConflict) {
			t.Fatalf("lapsed lease CheckpointEntryResult: expected ErrConflict, got %v", err)
		}
	})

	t.Run("terminal batch fences entry writes even with ownership untouched", func(t *testing.T) {
		const id = "batch-terminal-owned"
		epoch := env.activate(t, id)

		// Terminalize the row directly, the way a finalizer commit does,
		// without clearing processor_id/epoch/owner_lease_expires_at/resumable.
		// A late write from the batch's own (still-recorded) owner must still
		// be rejected: mirrors MockResumableStore.ownedFence.
		if _, err := env.pool.Exec(env.ctx,
			`UPDATE batch_items SET status = '{"status":"completed"}'::jsonb WHERE id = $1`, id,
		); err != nil {
			t.Fatalf("terminalize: %v", err)
		}

		if _, err := env.store.MarkEntryInFlight(env.ctx, id, "batch_req_b", testProcessorID, epoch); !errors.Is(err, api.ErrConflict) {
			t.Fatalf("terminal batch MarkEntryInFlight: expected ErrConflict, got %v", err)
		}
		_, err := env.store.CheckpointEntryResult(env.ctx, id, "batch_req_a", testProcessorID, epoch,
			&api.ResumableResult{Status: api.ResumableEntryCompleted,
				ResultKey: "results/j1/batch_req_a.jsonl", ResultBytes: 10, ResultSHA256: "sha-a"})
		if !errors.Is(err, api.ErrConflict) {
			t.Fatalf("terminal batch CheckpointEntryResult: expected ErrConflict, got %v", err)
		}
	})

	t.Run("rejects non-terminal checkpoint statuses", func(t *testing.T) {
		for _, status := range []api.ResumableEntryStatus{api.ResumableEntryPending, api.ResumableEntryInFlight} {
			committed, err := env.store.CheckpointEntryResult(env.ctx, batchID, "batch_req_b", testProcessorID, epoch,
				&api.ResumableResult{Status: status, ResultKey: "k", ResultBytes: 1, ResultSHA256: "s"})
			if err == nil || committed {
				t.Fatalf("status %q: expected a validation error, got committed=%v err=%v", status, committed, err)
			}
		}
		entry, ferr := env.store.FindEntryByRequestID(env.ctx, batchID, "batch_req_b")
		if ferr != nil || entry == nil || entry.Status != api.ResumableEntryPending || entry.Attempts != 0 {
			t.Fatalf("rejected checkpoints must not touch the entry: %v %+v", ferr, entry)
		}
	})

	t.Run("entry writes serialize with a concurrent requeue of the batch row", func(t *testing.T) {
		const id = "batch-entry-lock"
		epoch := env.activate(t, id)

		conn, err := env.pool.Acquire(env.ctx)
		if err != nil {
			t.Fatalf("acquire conn: %v", err)
		}
		defer conn.Release()
		tx, err := conn.Begin(env.ctx)
		if err != nil {
			t.Fatalf("begin: %v", err)
		}
		// The requeue's UPDATE, held uncommitted so it owns the batch row lock.
		if _, err := tx.Exec(env.ctx,
			`UPDATE batch_items
			 SET processor_id = NULL, owner_lease_expires_at = NULL,
			     epoch = epoch + 1, recovery_attempts = recovery_attempts + 1
			 WHERE id = $1`, id,
		); err != nil {
			t.Fatalf("lock batch row: %v", err)
		}

		type outcome struct {
			committed bool
			err       error
		}
		done := make(chan outcome, 1)
		go func() {
			committed, err := env.store.CheckpointEntryResult(env.ctx, id, "batch_req_a", testProcessorID, epoch,
				&api.ResumableResult{Status: api.ResumableEntryCompleted, ResultKey: "k", ResultBytes: 1, ResultSHA256: "s"})
			done <- outcome{committed, err}
		}()

		// The checkpoint must block on the batch-row lock: committing against
		// the pre-requeue snapshot would be exactly the fenced-out stale write.
		select {
		case o := <-done:
			t.Fatalf("checkpoint completed while the requeue held the batch row: committed=%v err=%v", o.committed, o.err)
		case <-time.After(500 * time.Millisecond):
		}

		if err := tx.Commit(env.ctx); err != nil {
			t.Fatalf("commit requeue: %v", err)
		}

		select {
		case o := <-done:
			if !errors.Is(o.err, api.ErrConflict) {
				t.Fatalf("expected ErrConflict after the requeue committed, got committed=%v err=%v", o.committed, o.err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("checkpoint did not return after the requeue committed")
		}
	})
}

func TestResumableEntryQueriesPG(t *testing.T) {
	env := newResumablePgEnv(t)

	const batchID = "batch-queries"
	epoch := env.activate(t, batchID)
	if _, err := env.store.CheckpointEntryResult(env.ctx, batchID, "batch_req_a", testProcessorID, epoch,
		&api.ResumableResult{Status: api.ResumableEntryCompleted, ResultKey: "k-a", ResultBytes: 1, ResultSHA256: "s-a"}); err != nil {
		t.Fatalf("checkpoint a: %v", err)
	}
	if _, err := env.store.MarkEntryInFlight(env.ctx, batchID, "batch_req_b", testProcessorID, epoch); err != nil {
		t.Fatalf("mark b: %v", err)
	}

	t.Run("resume state aggregates counts and terminal IDs", func(t *testing.T) {
		state, err := env.store.EntryResumeState(env.ctx, batchID)
		if err != nil {
			t.Fatalf("EntryResumeState: %v", err)
		}
		if state.Pending != 0 || state.InFlight != 1 || state.Completed != 1 || state.Failed != 1 {
			t.Fatalf("unexpected counts: %+v", state)
		}
		if len(state.TerminalRequestIDs) != 2 ||
			state.TerminalRequestIDs[0] != "batch_req_a" || state.TerminalRequestIDs[1] != "batch_req_r" {
			t.Fatalf("unexpected terminal IDs: %+v", state.TerminalRequestIDs)
		}
	})

	t.Run("list paginates by ordinal", func(t *testing.T) {
		first, err := env.store.ListBatchEntries(env.ctx, batchID, -1, 2)
		if err != nil || len(first) != 2 || first[0].Ordinal != 0 || first[1].Ordinal != 1 {
			t.Fatalf("first page: %v %+v", err, first)
		}
		second, err := env.store.ListBatchEntries(env.ctx, batchID, first[1].Ordinal, 2)
		if err != nil || len(second) != 1 || second[0].RequestID != "batch_req_r" {
			t.Fatalf("second page: %v %+v", err, second)
		}
	})

	t.Run("find by request ID", func(t *testing.T) {
		entry, err := env.store.FindEntryByRequestID(env.ctx, batchID, "batch_req_b")
		if err != nil || entry == nil || entry.CustomID != "b" || entry.Status != api.ResumableEntryInFlight {
			t.Fatalf("FindEntryByRequestID: %v %+v", err, entry)
		}
		missing, err := env.store.FindEntryByRequestID(env.ctx, batchID, "batch_req_missing")
		if err != nil || missing != nil {
			t.Fatalf("missing entry: %v %+v", err, missing)
		}
	})

	t.Run("entries cascade on batch delete and enforce key uniqueness", func(t *testing.T) {
		if _, err := env.pool.Exec(env.ctx,
			`INSERT INTO batch_entries (batch_id, request_id, ordinal, custom_id)
			 VALUES ($1, 'batch_req_a', 99, 'dup')`, batchID,
		); err == nil {
			t.Fatal("expected primary key violation for duplicate (batch_id, request_id)")
		}

		if _, err := env.pool.Exec(env.ctx, `DELETE FROM batch_items WHERE id = $1`, batchID); err != nil {
			t.Fatalf("delete batch: %v", err)
		}
		var n int
		if err := env.pool.QueryRow(env.ctx, `SELECT COUNT(*) FROM batch_entries WHERE batch_id = $1`, batchID).Scan(&n); err != nil {
			t.Fatalf("count entries: %v", err)
		}
		if n != 0 {
			t.Fatalf("entries must cascade with the batch row, %d remain", n)
		}
	})
}
