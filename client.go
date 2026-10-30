package aimix

import (
	"context"
	"fmt"
)

// Client routes requests to the provider named by Model.Provider.
type Client struct {
	providers map[ProviderID]Provider
}

// NewClient returns a Client serving the given providers. A later provider with
// the same ID replaces an earlier one.
func NewClient(providers ...Provider) *Client {
	c := &Client{providers: make(map[ProviderID]Provider, len(providers))}
	for _, p := range providers {
		c.providers[p.ID()] = p
	}
	return c
}

// Provider returns the provider registered under id.
func (c *Client) Provider(id ProviderID) (Provider, bool) {
	p, ok := c.providers[id]
	return p, ok
}

// Stream normalizes conv and streams one assistant turn with provider-specific
// defaults (thinking off unless the model requires it).
func (c *Client) Stream(ctx context.Context, model Model, conv Context, opts StreamOptions) (*AssistantMessageEventStream, error) {
	p, err := c.lookup(model)
	if err != nil {
		return nil, err
	}
	return p.Stream(ctx, StreamRequest{
		Model:   model,
		Context: NormalizeContext(conv),
		Options: opts,
	})
}

// StreamSimple is Stream with a provider-neutral thinking level.
func (c *Client) StreamSimple(ctx context.Context, model Model, conv Context, opts SimpleStreamOptions) (*AssistantMessageEventStream, error) {
	p, err := c.lookup(model)
	if err != nil {
		return nil, err
	}
	return p.StreamSimple(ctx, SimpleStreamRequest{
		Model:   model,
		Context: NormalizeContext(conv),
		Options: opts,
	})
}

// Complete runs Stream and waits for the final message. See Collect.
func (c *Client) Complete(ctx context.Context, model Model, conv Context, opts StreamOptions) (AssistantMessage, error) {
	s, err := c.Stream(ctx, model, conv, opts)
	if err != nil {
		return AssistantMessage{}, err
	}
	return Collect(ctx, s)
}

// CompleteSimple runs StreamSimple and waits for the final message. See Collect.
func (c *Client) CompleteSimple(ctx context.Context, model Model, conv Context, opts SimpleStreamOptions) (AssistantMessage, error) {
	s, err := c.StreamSimple(ctx, model, conv, opts)
	if err != nil {
		return AssistantMessage{}, err
	}
	return Collect(ctx, s)
}

func (c *Client) lookup(model Model) (Provider, error) {
	p, ok := c.providers[model.Provider]
	if !ok {
		return nil, fmt.Errorf("aimix: no provider registered for %q", model.Provider)
	}
	return p, nil
}

// StreamError reports an assistant turn that ended with StopReasonError or
// StopReasonAborted. The message returned alongside it keeps any partial content.
type StreamError struct {
	Reason  StopReason
	Message string
	// Err is the underlying error when the stream delivered it.
	Err error
}

func (e *StreamError) Error() string {
	if e.Message != "" {
		return e.Message
	}
	return "aimix: stream ended with " + string(e.Reason)
}

func (e *StreamError) Unwrap() error {
	return e.Err
}

// Collect drains s and returns the final message. A turn that ended with an
// error or was aborted is returned together with a *StreamError.
func Collect(ctx context.Context, s *AssistantMessageEventStream) (AssistantMessage, error) {
	var cause error
	for ev := range s.Events() {
		if e, ok := ev.(StreamErrorEvent); ok {
			cause = e.Err
		}
	}
	msg, err := s.Result(ctx)
	if err != nil {
		return msg, err
	}
	if msg.StopReason == StopReasonError || msg.StopReason == StopReasonAborted {
		return msg, &StreamError{
			Reason:  msg.StopReason,
			Message: msg.ErrorReason,
			Err:     cause,
		}
	}
	return msg, nil
}
