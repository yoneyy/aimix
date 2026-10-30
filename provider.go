package aimix

import "context"

type ProviderID string

func (p ProviderID) String() string {
	return string(p)
}

const (
	ProviderOpenAI    ProviderID = "openai"
	ProviderAnthropic ProviderID = "anthropic"
)

type StreamRequest struct {
	Model   Model
	Context TranscriptContext
	Options StreamOptions
}

type SimpleStreamRequest struct {
	Model   Model
	Context TranscriptContext
	Options SimpleStreamOptions
}

// ApiStreamer is the implementation of one Api.
//
// Stream and StreamSimple return an error only for problems found before any
// request is sent, such as a missing API key. Everything after that, including
// HTTP errors and cancellation, ends the stream with a StreamErrorEvent.
//
// Callers must drain Events until it is closed, or cancel ctx; the producer
// blocks while the event buffer is full.
type ApiStreamer interface {
	Stream(ctx context.Context, req StreamRequest) (*AssistantMessageEventStream, error)
	StreamSimple(ctx context.Context, req SimpleStreamRequest) (*AssistantMessageEventStream, error)
}

// Provider is one vendor: it resolves the API key and base URL for a model and
// dispatches to the ApiStreamer of the model's Api.
type Provider interface {
	ID() ProviderID
	Name() string
	ApiStreamer
}
