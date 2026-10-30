package aimix

import "encoding/json"

type ContentType string

const (
	ContentTypeText     ContentType = "text"
	ContentTypeThinking ContentType = "thinking"
	ContentTypeImage    ContentType = "image"
	ContentTypeToolCall ContentType = "tool_call"
)

type Content interface {
	implContent()
}

type TextContent struct {
	Type          ContentType `json:"type"`
	Text          string      `json:"text"`
	TextSignature string      `json:"text_signature,omitempty"` // e.g., for OpenAI responses, message metadata (legacy id string or TextSignatureV1 JSON)
}

type ThinkingContent struct {
	Type              ContentType `json:"type"`
	Thinking          string      `json:"thinking"`
	ThinkingSignature string      `json:"thinking_signature,omitempty"` // Provider-specific opaque or serialized reasoning replay data
	// When true, the thinking content was redacted by safety filters. The opaque
	// encrypted payload is stored in ThinkingSignature so it can be passed back
	// to the API for multi-turn continuity.
	Redacted bool `json:"redacted,omitempty"`
}

type ImageContent struct {
	Type     ContentType `json:"type"`
	Data     string      `json:"data"`      // base64 encoded image data
	MimeType string      `json:"mime_type"` // e.g., "image/jpeg", "image/png"
}

type ToolCallContent struct {
	Type      ContentType     `json:"type"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	// Google-specific: opaque signature for reusing thought context
	ThoughtSignature string `json:"thought_signature,omitempty"`
	// OpenAI Responses namespace for calls to dynamically loaded or namespaced tools.
	Namespace string `json:"namespace,omitempty"`
	// Set when the model's complete arguments text was not a valid JSON object. `arguments` is then `{}`.
	// Describes the problem and includes the text the model sent; tool argument validation fails with it.
	ArgumentsError string `json:"arguments_error,omitempty"`
}

func (TextContent) implContent()     {}
func (ThinkingContent) implContent() {}
func (ImageContent) implContent()    {}
func (ToolCallContent) implContent() {}

type Contents []Content
