package aimix

import (
	"encoding/json"
	"time"
)

type Usage struct {
	// Input tokens, excluding CacheRead and CacheWrite
	Input int `json:"input"`
	// Output tokens
	Output     int `json:"output"`
	CacheRead  int `json:"cache_read"`
	CacheWrite int `json:"cache_write"`
	//  Subset of `cacheWrite` written with 1h retention. Only Anthropic reports this split.
	CacheWrite1h int `json:"cache_write_1h"`
	// Reasoning/thinking tokens, when the provider reports them. This is a subset of
	// `output`: `output` already includes these tokens. Set to a number (possibly 0) by
	// providers that expose a reasoning breakdown; left undefined by providers that don't.
	Reasoning int `json:"reasoning"`
	// Input + Output + CacheRead + CacheWrite
	TotalTokens int `json:"total_tokens"`
}

// SystemMessage carries instructions and tool set changes. The leading one holds the
// base prompt and initial tools (see NormalizeContext); later ones add instructions and
// change the tool set from that point on. Replaying every system message in order
// yields the current prompt and tools.
type SystemMessage struct {
	Role    Role          `json:"role"`
	Content []TextContent `json:"content,omitempty"`
	// ToolsAdded holds complete definitions of tools that become available at this point.
	ToolsAdded []Tool `json:"tools_added,omitempty"`
	// ToolsRemoved names tools that stop being available at this point.
	ToolsRemoved []string `json:"tools_removed,omitempty"`

	Timestamp int64 `json:"timestamp"` // Unix timestamp in milliseconds
}

type UserMessage struct {
	Role      Role     `json:"role"`
	Content   Contents `json:"content"`
	Timestamp int64    `json:"timestamp"` // Unix timestamp in milliseconds
}

type AssistantMessage struct {
	Role          Role          `json:"role"`
	Content       Contents      `json:"content"`
	Api           Api           `json:"api"`
	Provider      ProviderID    `json:"provider"`
	Model         string        `json:"model"`
	ResponseModel string        `json:"response_model,omitempty"` // Concrete model reported by the provider, when reported. May equal the requested Model
	ResponseID    string        `json:"response_id,omitempty"`    // Provider-specific response/message identifier when the upstream API exposes one
	ThinkingLevel ThinkingLevel `json:"thinking_level,omitempty"`
	Usage         Usage         `json:"usage"`
	StopReason    StopReason    `json:"stop_reason"`
	RawStopReason string        `json:"raw_stop_reason,omitempty"` // Stop reason exactly as the provider reported it
	ErrorReason   string        `json:"error_reason,omitempty"`
	Timestamp     int64         `json:"timestamp"` // Unix timestamp in milliseconds
}

type ToolResultMessage struct {
	Role       Role            `json:"role"`
	ToolCallID string          `json:"tool_call_id"`
	ToolName   string          `json:"tool_name"`
	Content    Contents        `json:"content"`
	Details    json.RawMessage `json:"details,omitempty"`
	Usage      Usage           `json:"usage"`
	IsError    bool            `json:"is_error"`
	Timestamp  int64           `json:"timestamp"` // Unix timestamp in milliseconds
	Duration   time.Duration   `json:"duration"`  // Duration Time consumed by tool execution
}

// Message is a sealed interface: only types in this package can implement it.
type Message interface {
	implMessage()
	GetRole() Role
}

// impl Message
func (SystemMessage) implMessage()     {}
func (UserMessage) implMessage()       {}
func (AssistantMessage) implMessage()  {}
func (ToolResultMessage) implMessage() {}

func (m SystemMessage) GetRole() Role     { return RoleSystem }
func (m UserMessage) GetRole() Role       { return RoleUser }
func (m AssistantMessage) GetRole() Role  { return RoleAssistant }
func (m ToolResultMessage) GetRole() Role { return RoleToolResult }

// StopReason is the provider-neutral reason an assistant turn ended. The value
// the provider reported is kept in AssistantMessage.RawStopReason.
type StopReason string

func (s StopReason) String() string {
	return string(s)
}

const (
	// StopReasonStop means the model finished its turn normally.
	StopReasonStop StopReason = "stop"
	// StopReasonLength means the output hit the max tokens limit.
	StopReasonLength StopReason = "length"
	// StopReasonToolUse means the model stopped to wait for tool results.
	StopReasonToolUse StopReason = "tool_use"
	// StopReasonError means the request or the stream failed; see ErrorReason.
	StopReasonError StopReason = "error"
	// StopReasonAborted means the caller canceled the context.
	StopReasonAborted StopReason = "aborted"
)
