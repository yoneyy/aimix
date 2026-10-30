package api

import (
	"encoding/json"
	"testing"

	"github.com/yoneyy/aimix"
)

func TestParallelToolCalls(t *testing.T) {
	withTools := aimix.NormalizeContext(aimix.Context{
		Tools: []aimix.Tool{
			{
				Name:       "t",
				Parameters: json.RawMessage(`{"type":"object"}`),
			},
		},
	})
	noTools := aimix.TranscriptContext{}

	// body renders the request params and returns the given top-level fields,
	// "absent" for missing ones.
	body := func(t *testing.T, params any, fields ...string) map[string]string {
		t.Helper()
		var m map[string]any
		if err := json.Unmarshal([]byte(compactJSON(t, params)), &m); err != nil {
			t.Fatal(err)
		}
		out := make(map[string]string, len(fields))
		for _, f := range fields {
			out[f] = "absent"
			if v, ok := m[f]; ok {
				out[f] = compactJSON(t, v)
			}
		}
		return out
	}

	tests := []struct {
		name     string
		tc       aimix.TranscriptContext
		parallel *bool
		choice   aimix.ToolChoice
		// Anthropic tool_choice, OpenAI parallel_tool_calls
		anthropic string
		openai    string
	}{
		{
			name:      "nil sends nothing",
			tc:        withTools,
			anthropic: "absent",
			openai:    "absent",
		},
		{
			name:      "true",
			tc:        withTools,
			parallel:  ptr(true),
			anthropic: "absent",
			openai:    "true",
		},
		{
			name:      "false",
			tc:        withTools,
			parallel:  ptr(false),
			anthropic: `{"disable_parallel_tool_use":true,"type":"auto"}`,
			openai:    "false",
		},
		{
			name:      "false with required",
			tc:        withTools,
			parallel:  ptr(false),
			choice:    aimix.ToolChoiceRequired,
			anthropic: `{"disable_parallel_tool_use":true,"type":"any"}`,
			openai:    "false",
		},
		{
			name:      "false without tools",
			tc:        noTools,
			parallel:  ptr(false),
			anthropic: "absent",
			openai:    "absent",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := aimix.StreamOptions{
				ParallelToolCalls: tt.parallel,
				ToolChoice:        tt.choice,
			}

			ap, _, err := buildAnthropicParams(anthropicModel(""), tt.tc, AnthropicOptions{
				StreamOptions: opts,
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := body(t, ap, "tool_choice")["tool_choice"]; got != tt.anthropic {
				t.Errorf("anthropic tool_choice %s, want %s", got, tt.anthropic)
			}

			rp, err := buildResponsesParams(responsesModel(""), tt.tc, OpenAIResponsesOptions{
				StreamOptions: opts,
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := body(t, rp, "parallel_tool_calls")["parallel_tool_calls"]; got != tt.openai {
				t.Errorf("responses parallel_tool_calls %s, want %s", got, tt.openai)
			}

			cp, err := buildCompletionsParams(completionsModel(""), tt.tc, OpenAICompletionsOptions{
				StreamOptions: opts,
			})
			if err != nil {
				t.Fatal(err)
			}
			if got := body(t, cp, "parallel_tool_calls")["parallel_tool_calls"]; got != tt.openai {
				t.Errorf("completions parallel_tool_calls %s, want %s", got, tt.openai)
			}
		})
	}
}
