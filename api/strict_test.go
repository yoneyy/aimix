package api

import (
	"encoding/json"
	"testing"

	"github.com/yoneyy/aimix"
)

// TestToolStrict checks how Tool.Strict reaches each API: Responses always sends
// the flag (the API defaults it to true), the others only send it when set.
func TestToolStrict(t *testing.T) {
	tc := func(strict bool) aimix.TranscriptContext {
		return aimix.NormalizeContext(aimix.Context{
			Tools: []aimix.Tool{
				{
					Name:       "t",
					Parameters: json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
					Strict:     strict,
				},
			},
		})
	}
	// strictOf returns the "strict" value of the first tool, or "absent".
	strictOf := func(t *testing.T, params any, path ...string) string {
		t.Helper()
		var body map[string]any
		if err := json.Unmarshal([]byte(compactJSON(t, params)), &body); err != nil {
			t.Fatal(err)
		}
		tool := body["tools"].([]any)[0].(map[string]any)
		for _, p := range path {
			tool = tool[p].(map[string]any)
		}
		v, ok := tool["strict"]
		if !ok {
			return "absent"
		}
		return compactJSON(t, v)
	}

	tests := []struct {
		name  string
		build func(strict bool) (any, error)
		path  []string
		on    string
		off   string
	}{
		{
			name: "anthropic",
			build: func(strict bool) (any, error) {
				p, _, err := buildAnthropicParams(anthropicModel(""), tc(strict), AnthropicOptions{})
				return p, err
			},
			on:  "true",
			off: "absent",
		},
		{
			name: "responses",
			build: func(strict bool) (any, error) {
				return buildResponsesParams(responsesModel(""), tc(strict), OpenAIResponsesOptions{})
			},
			on:  "true",
			off: "false",
		},
		{
			name: "completions",
			build: func(strict bool) (any, error) {
				return buildCompletionsParams(completionsModel(""), tc(strict), OpenAICompletionsOptions{})
			},
			path: []string{"function"},
			on:   "true",
			off:  "absent",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			for strict, want := range map[bool]string{true: tt.on, false: tt.off} {
				params, err := tt.build(strict)
				if err != nil {
					t.Fatal(err)
				}
				if got := strictOf(t, params, tt.path...); got != want {
					t.Errorf("Strict=%v: strict %s, want %s", strict, got, want)
				}
			}
		})
	}
}
