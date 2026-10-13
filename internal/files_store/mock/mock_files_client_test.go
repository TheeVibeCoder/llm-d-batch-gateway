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

package mock

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/llm-d/llm-d-batch-gateway/internal/files_store/api"
)

func newTestClient(t *testing.T) *MockBatchFilesClient {
	t.Helper()
	return NewMockBatchFilesClient(t.TempDir())
}

func TestStore(t *testing.T) {
	ctx := context.Background()

	t.Run("stores file successfully", func(t *testing.T) {
		client := newTestClient(t)
		content := []byte("line1\nline2\n")

		md, err := client.Store(ctx, "out.jsonl", "folder", 0, 0, bytes.NewReader(content))
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if md.Size != int64(len(content)) || md.LinesNumber != 2 {
			t.Errorf("unexpected metadata: %+v", md)
		}

		data, err := os.ReadFile(filepath.Join(client.rootDir, "folder", "out.jsonl"))
		if err != nil {
			t.Fatalf("failed to read stored file: %v", err)
		}
		if !bytes.Equal(data, content) {
			t.Errorf("expected content %q, got %q", content, data)
		}
	})

	t.Run("returns ErrFileExists for stored file", func(t *testing.T) {
		client := newTestClient(t)
		content := []byte("line1\n")

		if _, err := client.Store(ctx, "out.jsonl", "folder", 0, 0, bytes.NewReader(content)); err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if _, err := client.Store(ctx, "out.jsonl", "folder", 0, 0, bytes.NewReader(content)); !errors.Is(err, api.ErrFileExists) {
			t.Errorf("expected ErrFileExists, got %v", err)
		}
	})

	t.Run("failed store can be retried", func(t *testing.T) {
		client := newTestClient(t)

		_, err := client.Store(ctx, "out.jsonl", "folder", 5, 0, strings.NewReader("too large"))
		if err == nil {
			t.Fatal("expected size limit error")
		}
		// The failed attempt must not leave a file behind, so the retry
		// with a larger limit succeeds.
		if _, err := client.Store(ctx, "out.jsonl", "folder", 1024, 0, strings.NewReader("too large")); err != nil {
			t.Fatalf("retry after failed store: %v", err)
		}
	})
}

func TestOpenStream(t *testing.T) {
	ctx := context.Background()

	t.Run("commit writes the buffered data", func(t *testing.T) {
		client := newTestClient(t)
		content := []byte("line1\nline2\n")

		w, err := client.OpenStream(ctx, "out.jsonl", "folder")
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if _, err := w.Write(content); err != nil {
			t.Fatalf("write failed: %v", err)
		}

		md, err := w.Commit(ctx)
		if err != nil {
			t.Fatalf("commit failed: %v", err)
		}
		if md.Size != int64(len(content)) {
			t.Errorf("expected size %d, got %d", len(content), md.Size)
		}

		data, err := os.ReadFile(filepath.Join(client.rootDir, "folder", "out.jsonl"))
		if err != nil {
			t.Fatalf("failed to read committed file: %v", err)
		}
		if !bytes.Equal(data, content) {
			t.Errorf("expected content %q, got %q", content, data)
		}
	})

	t.Run("abort writes nothing", func(t *testing.T) {
		client := newTestClient(t)

		w, err := client.OpenStream(ctx, "out.jsonl", "folder")
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if _, err := w.Write([]byte("discarded")); err != nil {
			t.Fatalf("write failed: %v", err)
		}
		if err := w.Abort(ctx); err != nil {
			t.Fatalf("abort failed: %v", err)
		}

		if _, err := os.Stat(filepath.Join(client.rootDir, "folder", "out.jsonl")); !os.IsNotExist(err) {
			t.Error("expected no file after abort")
		}
	})

	t.Run("returns ErrFileExists for stored file", func(t *testing.T) {
		client := newTestClient(t)

		w, err := client.OpenStream(ctx, "out.jsonl", "folder")
		if err != nil {
			t.Fatalf("expected no error, got %v", err)
		}
		if _, err := w.Commit(ctx); err != nil {
			t.Fatalf("commit failed: %v", err)
		}

		if _, err := client.OpenStream(ctx, "out.jsonl", "folder"); !errors.Is(err, api.ErrFileExists) {
			t.Errorf("expected ErrFileExists, got %v", err)
		}
	})
}
