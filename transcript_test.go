package aimix

import (
	"encoding/json"
	"reflect"
	"testing"
)

func tool(name, desc string) Tool {
	return Tool{
		Name:        name,
		Description: desc,
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
}

func TestNormalizeContext(t *testing.T) {
	user := UserMessage{
		Role: RoleUser,
	}
	if got := NormalizeContext(Context{Messages: Messages{user}}); len(got.Messages) != 1 {
		t.Fatalf("empty prompt added a system message: %+v", got.Messages)
	}

	got := NormalizeContext(Context{
		SystemPrompt: "prompt",
		Messages:     Messages{user},
		Tools: []Tool{
			tool("a", ""),
		},
	})
	head, ok := AsSystemMessage(got.Messages[0])
	if !ok || TextOf(head.Content) != "prompt" || len(head.ToolsAdded) != 1 || len(got.Messages) != 2 {
		t.Fatalf("got %+v", got.Messages)
	}
}

func TestCurrentToolsAndCollapse(t *testing.T) {
	messages := Messages{
		SystemMessage{
			Role: RoleSystem,
			Content: []TextContent{
				{
					Type: ContentTypeText,
					Text: "base",
				},
			},
			ToolsAdded: []Tool{
				tool("a", "v1"),
				tool("b", ""),
			},
			Timestamp: 7,
		},
		UserMessage{
			Role: RoleUser,
		},
		&SystemMessage{
			Role: RoleSystem,
			Content: []TextContent{
				{
					Type: ContentTypeText,
					Text: "more",
				},
			},
			ToolsRemoved: []string{
				"b",
			},
			ToolsAdded: []Tool{
				tool("c", ""),
				tool("a", "v2"),
			},
		},
	}

	var names, descs []string
	for _, tl := range CurrentTools(messages) {
		names = append(names, tl.Name)
		descs = append(descs, tl.Description)
	}
	// A redefined tool keeps its position and takes the new definition.
	if !reflect.DeepEqual(names, []string{"a", "c"}) || descs[0] != "v2" {
		t.Fatalf("tools %v %v", names, descs)
	}

	collapsed := CollapseSystemMessages(TranscriptContext{Messages: messages})
	if len(collapsed.Messages) != 2 {
		t.Fatalf("collapsed %+v", collapsed.Messages)
	}
	head := collapsed.Messages[0].(SystemMessage)
	if TextOf(head.Content) != "base\n\nmore" || len(head.ToolsAdded) != 2 || head.Timestamp != 7 {
		t.Fatalf("head %+v", head)
	}
	if _, ok := collapsed.Messages[1].(UserMessage); !ok {
		t.Fatalf("rest %+v", collapsed.Messages[1])
	}
}

func TestMessagesJSONRoundTrip(t *testing.T) {
	in := Messages{
		SystemMessage{
			Content: []TextContent{
				{
					Text: "sys",
				},
			},
		},
		UserMessage{
			Content: Contents{
				TextContent{
					Text: "hi",
				},
				ImageContent{
					Data:     "AAAA",
					MimeType: "image/png",
				},
			},
		},
		AssistantMessage{
			Api:        ApiAnthropicMessages,
			Provider:   ProviderAnthropic,
			Model:      "claude",
			StopReason: StopReasonToolUse,
			Content: Contents{
				ThinkingContent{
					Thinking:          "hmm",
					ThinkingSignature: "sig",
				},
				&ToolCallContent{
					ID:   "call_1",
					Name: "read",
				},
			},
		},
		ToolResultMessage{
			ToolCallID: "call_1",
			ToolName:   "read",
			IsError:    true,
			Content: Contents{
				TextContent{
					Text: "boom",
				},
			},
		},
	}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out Messages
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatalf("unmarshal %s: %v", raw, err)
	}

	if len(out) != 4 {
		t.Fatalf("got %d messages", len(out))
	}
	if m := out[0].(SystemMessage); m.Role != RoleSystem || m.Content[0].Type != ContentTypeText {
		t.Fatalf("system %+v", m)
	}
	user := out[1].(UserMessage)
	if user.Role != RoleUser || user.Content[1].(ImageContent).MimeType != "image/png" {
		t.Fatalf("user %+v", user)
	}
	asst := out[2].(AssistantMessage)
	call := asst.Content[1].(ToolCallContent)
	if asst.Role != RoleAssistant || asst.StopReason != StopReasonToolUse || call.Type != ContentTypeToolCall || string(call.Arguments) != "{}" {
		t.Fatalf("assistant %+v", asst)
	}
	if tr := out[3].(ToolResultMessage); tr.Role != RoleToolResult || !tr.IsError || tr.Content[0].(TextContent).Text != "boom" {
		t.Fatalf("tool result %+v", tr)
	}

	var ctx Context
	if err := json.Unmarshal([]byte(`{"messages":[{"role":"user","content":[{"type":"text","text":"x"}]}]}`), &ctx); err != nil {
		t.Fatal(err)
	}
	if _, ok := ctx.Messages[0].(UserMessage); !ok {
		t.Fatalf("context messages %+v", ctx.Messages)
	}
	if err := json.Unmarshal([]byte(`[{"role":"robot"}]`), &out); err == nil {
		t.Fatal("unknown role accepted")
	}
}

func TestClampThinkingLevel(t *testing.T) {
	plain := Model{}
	reasoning := Model{
		Reasoning: true,
	}
	mapped := Model{
		Reasoning: true,
		ThinkingLevels: ThinkingLevelMap{
			ThinkingOff:     "",
			ThinkingMinimal: "",
			ThinkingMax:     "max",
		},
	}

	tests := []struct {
		model Model
		in    ThinkingLevel
		want  ThinkingLevel
	}{
		{model: plain, in: ThinkingHigh, want: ThinkingOff},
		{model: reasoning, in: ThinkingHigh, want: ThinkingHigh},
		{model: reasoning, in: ThinkingXhigh, want: ThinkingHigh},
		{model: reasoning, in: ThinkingMax, want: ThinkingHigh},
		{model: mapped, in: ThinkingOff, want: ThinkingLow},
		{model: mapped, in: ThinkingXhigh, want: ThinkingMax},
		{model: mapped, in: "bogus", want: ThinkingLow},
	}
	for _, tt := range tests {
		if got := ClampThinkingLevel(tt.model, tt.in); got != tt.want {
			t.Errorf("ClampThinkingLevel(%+v, %s) = %s, want %s", tt.model.ThinkingLevels, tt.in, got, tt.want)
		}
	}
	want := []ThinkingLevel{
		ThinkingLow,
		ThinkingMedium,
		ThinkingHigh,
		ThinkingMax,
	}
	if got := SupportedThinkingLevels(mapped); !reflect.DeepEqual(got, want) {
		t.Fatalf("supported %v", got)
	}
}
