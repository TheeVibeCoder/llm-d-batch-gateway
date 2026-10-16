package pipeline

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/go-logr/logr"

	"github.com/llm-d/llm-d-batch-gateway/internal/shared/syncutil"
	batch_types "github.com/llm-d/llm-d-batch-gateway/internal/shared/types"
	"github.com/llm-d/llm-d-batch-gateway/pkg/clients/inference"
)

const defaultResultBuffer = 64

// BroadcasterGroup is a set of broadcasters for subscribe/unsubscribe.
// Lifecycle (Run/Wait) is owned by the registry that created them.
type BroadcasterGroup struct {
	broadcasters []*ResultBroadcaster
}

func NewBroadcasterGroup(broadcasters []*ResultBroadcaster) *BroadcasterGroup {
	return &BroadcasterGroup{broadcasters: broadcasters}
}

func (bs *BroadcasterGroup) Subscribe(ch chan<- ResultItem) {
	for _, b := range bs.broadcasters {
		b.Subscribe(ch)
	}
}

func (bs *BroadcasterGroup) Unsubscribe(ch chan<- ResultItem) {
	for _, b := range bs.broadcasters {
		b.Unsubscribe(ch)
	}
}

// ResultBroadcaster reads results from a shared async client and
// broadcasts to all subscribed channels.
type ResultBroadcaster struct {
	client      inference.AsyncInferenceClient
	subscribers *syncutil.MutexMap[chan<- ResultItem, struct{}]
	logger      logr.Logger
}

func NewResultBroadcaster(client inference.AsyncInferenceClient, logger logr.Logger) *ResultBroadcaster {
	return &ResultBroadcaster{
		client:      client,
		subscribers: syncutil.NewMutexMap[chan<- ResultItem, struct{}](),
		logger:      logger,
	}
}

// Subscribe registers dest to receive all results.
// dest should be buffered to avoid blocking the broadcaster.
func (b *ResultBroadcaster) Subscribe(dest chan<- ResultItem) {
	b.subscribers.Store(dest, struct{}{})
}

// Unsubscribe removes dest from the broadcast list.
func (b *ResultBroadcaster) Unsubscribe(dest chan<- ResultItem) {
	b.subscribers.Delete(dest)
}

// receivedResult pairs one pulled result with the acknowledgement of its
// durable delivery (nil on destructive transports). The ack travels with the
// result so the consumer that accounts for it can acknowledge afterwards.
type receivedResult struct {
	resp *inference.GenerateResponse
	ack  func(ctx context.Context) error
}

// Run reads results and broadcasts to all subscribers.
func (b *ResultBroadcaster) Run(ctx context.Context) {
	incomingCh := make(chan receivedResult)

	go func() {
		defer close(incomingCh)
		backoff := 100 * time.Millisecond
		const maxBackoff = 10 * time.Second
		for ctx.Err() == nil {
			received, err := b.receive(ctx)
			if err != nil {
				if ctx.Err() != nil {
					return
				}
				b.logger.Error(err, "GetResult failed, retrying", "backoff", backoff)
				select {
				case <-time.After(backoff):
				case <-ctx.Done():
					return
				}
				backoff = min(backoff*2, maxBackoff)
				continue
			}
			backoff = 100 * time.Millisecond
			select {
			case incomingCh <- received:
			case <-ctx.Done():
				// Unrouted delivery: leave it unacked so lease expiry
				// redelivers it to whoever runs next.
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return

		case received, ok := <-incomingCh:
			if !ok {
				return
			}
			result := asyncResult(received.resp, b.logger)

			subscribers := b.subscribers.Keys()
			if len(subscribers) == 0 {
				// No job in this processor awaits results on this model, so the
				// delivery is residue — e.g. a late result for a batch that
				// terminalized while its request was in flight. Acknowledge it
				// instead of letting its lease bounce forever; jobs subscribe
				// before submitting, so a subscriberless route holds no result a
				// live job still wants.
				if received.ack != nil {
					if err := received.ack(ctx); err != nil {
						b.logger.Info("Acknowledging unowned result failed; its lease will redeliver it",
							"requestID", result.RequestID, "err", err)
					}
				}
				continue
			}
			result.Ack = received.ack
			for _, ch := range subscribers {
				b.safeChannelSend(result, ch)
			}
		}
	}
}

// receive pulls one result, preferring the durable leased delivery: the
// acknowledgement travels with the result so its consumer acknowledges only
// after durably accounting for it. Transports without the capability keep
// the destructive GetResult contract.
func (b *ResultBroadcaster) receive(ctx context.Context) (receivedResult, error) {
	if dc, ok := b.client.(inference.DurableResultClient); ok {
		resp, ack, err := dc.ReceiveResult(ctx)
		if err == nil || !errors.Is(err, inference.ErrDurableResultsUnavailable) {
			return receivedResult{resp: resp, ack: ack}, err
		}
	}
	resp, err := b.client.GetResult(ctx)
	return receivedResult{resp: resp}, err
}

// safeChannelSend sends to ch and recovers from the panic if the channel
// was closed concurrently (subscriber unsubscribed). The only operation in
// this function is a channel send, so the only possible panic is
// "send on closed channel".
func (b *ResultBroadcaster) safeChannelSend(result ResultItem, ch chan<- ResultItem) {
	defer func() {
		if r := recover(); r != nil {
			if err, ok := r.(error); ok && err.Error() == "send on closed channel" {
				b.logger.Info("Broadcast send recovered (subscriber unsubscribed)",
					"requestID", result.RequestID)
			} else {
				panic(r)
			}
		}
	}()
	ch <- result
}

func asyncResult(resp *inference.GenerateResponse, logger logr.Logger) ResultItem {
	result := ResultItem{
		RequestID: resp.RequestID,
	}

	if resp.IsNonHTTPFailure() {
		code := resp.ErrorCode
		if code == "" {
			code = "server_error"
		}
		msg := resp.ErrorMessage
		if msg == "" {
			msg = "async request failed"
		}
		result.Error = &OutputError{Code: code, Message: msg}
		return result
	}

	statusCode := resp.StatusCode
	if statusCode == 0 {
		// Legacy ResultMessage with only Payload — treat as HTTP 200.
		statusCode = 200
	}

	if len(resp.Response) == 0 {
		if statusCode >= 200 && statusCode < 300 {
			result.Error = &OutputError{Code: "server_error", Message: "async response has no body"}
			return result
		}
		// Non-2xx with empty body (e.g. 403 from auth): preserve status code.
		result.Response = &batch_types.ResponseData{
			StatusCode: statusCode,
			RequestID:  resp.RequestID,
			Body:       map[string]any{},
		}
		return result
	}

	var body map[string]any
	if err := json.Unmarshal(resp.Response, &body); err != nil {
		if statusCode >= 200 && statusCode < 300 {
			logger.Error(err, "Failed to unmarshal async response", "requestID", resp.RequestID)
			result.Error = &OutputError{
				Code:    "parse_error",
				Message: fmt.Sprintf("response body could not be parsed: %v", err),
			}
			return result
		}
		// Non-2xx with unparsable body: still preserve the HTTP status.
		body = map[string]any{
			"error": map[string]any{
				"message": string(resp.Response),
			},
		}
	}
	result.Response = &batch_types.ResponseData{
		StatusCode: statusCode,
		RequestID:  resp.RequestID,
		Body:       body,
	}
	return result
}
