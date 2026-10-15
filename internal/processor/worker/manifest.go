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
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// manifestEntry is the per-line metadata accumulated while preprocessing a
// batch. RequestIDs are stable (derived from batch + custom IDs), so the same
// input always yields the same manifest.
type manifestEntry struct {
	Ordinal      int64
	CustomID     string
	RequestID    string
	Rejected     bool
	ErrorCode    string
	ResultKey    string
	ResultBytes  int64
	ResultSHA256 string
}

// requestManifest is the in-memory entry list of a preprocessed batch: the
// input for resumable activation and the comparison set for recovery's
// integrity check. It is never persisted as a blob.
type requestManifest struct {
	Entries []manifestEntry
}

func (m *requestManifest) add(e manifestEntry) {
	m.Entries = append(m.Entries, e)
}

// rejectedEntry records a line rejected during ingestion (e.g. unregistered
// model), with the metadata of its error-object line.
func rejectedEntry(batchID string, ordinal int64, customID, requestID, errorCode string, errLine []byte) manifestEntry {
	sum := sha256.Sum256(errLine)
	return manifestEntry{
		Ordinal:      ordinal,
		CustomID:     customID,
		RequestID:    requestID,
		Rejected:     true,
		ErrorCode:    errorCode,
		ResultKey:    resumableResultKey(batchID, requestID),
		ResultBytes:  int64(len(errLine)),
		ResultSHA256: hex.EncodeToString(sum[:]),
	}
}

// resumableResultKey is the file-store key of one request's result object.
func resumableResultKey(batchID, requestID string) string {
	return fmt.Sprintf("results/%s/%s.jsonl", batchID, requestID)
}
