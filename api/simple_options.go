package api

import (
	"encoding/json"
	"math"
	"unicode/utf8"

	"github.com/yoneyy/aimix"
)

const (
	// contextSafetyTokens is kept free in the context window when capping max tokens.
	contextSafetyTokens = 4096
	// minAnswerTokens is always left for the answer when a thinking budget shares
	// the response ceiling.
	minAnswerTokens     = 1024
	charsPerToken       = 3.5
	estimatedImageChars = 4800
)

var defaultThinkingBudgets = map[aimix.ThinkingLevel]int{
	aimix.ThinkingMinimal: 1024,
	aimix.ThinkingLow:     2048,
	aimix.ThinkingMedium:  8192,
	aimix.ThinkingHigh:    16384,
}

// effectiveReasoning clamps the requested level to the model. Off and empty
// both come back as ThinkingOff.
func effectiveReasoning(model aimix.Model, level aimix.ThinkingLevel) aimix.ThinkingLevel {
	if level == "" {
		level = aimix.ThinkingOff
	}
	return aimix.ClampThinkingLevel(model, level)
}

// baseMaxTokens is the caller's cap or the model's, fitted into the context
// window. 0 means neither is known.
func baseMaxTokens(model aimix.Model, messages []aimix.Message, opts aimix.StreamOptions) int {
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 {
		maxTokens = model.MaxTokens
	}
	if maxTokens <= 0 {
		return 0
	}
	return clampMaxTokensToContext(model, messages, maxTokens)
}

// clampMaxTokensToContext caps maxTokens to what is left of the context window
// after the estimated transcript and a safety margin. The result is at least 1.
func clampMaxTokensToContext(model aimix.Model, messages []aimix.Message, maxTokens int) int {
	if model.ContextWindow <= 0 {
		return max(1, maxTokens)
	}
	available := model.ContextWindow - estimateContextTokens(messages) - contextSafetyTokens
	return min(maxTokens, max(1, available))
}

// thinkingBudgetForLevel returns the budget of level; xhigh and max use the high
// budget.
func thinkingBudgetForLevel(level aimix.ThinkingLevel, custom map[aimix.ThinkingLevel]int) int {
	if level == aimix.ThinkingXhigh || level == aimix.ThinkingMax {
		level = aimix.ThinkingHigh
	}
	if b, ok := custom[level]; ok {
		return b
	}
	return defaultThinkingBudgets[level]
}

// adjustMaxTokensForThinking makes room for a thinking budget. callerMax is the
// caller's cap (0 for none): the budget is added on top of it, up to the model
// cap. When the result cannot hold the budget, the budget shrinks to leave
// minAnswerTokens for the answer.
func adjustMaxTokensForThinking(callerMax, modelMax int, level aimix.ThinkingLevel, custom map[aimix.ThinkingLevel]int) (maxTokens, budget int) {
	budget = thinkingBudgetForLevel(level, custom)
	switch {
	case callerMax <= 0:
		maxTokens = modelMax
	case modelMax <= 0:
		maxTokens = callerMax + budget
	default:
		maxTokens = min(callerMax+budget, modelMax)
	}
	if maxTokens <= budget {
		budget = min(budget, max(0, maxTokens-minAnswerTokens))
	}
	return maxTokens, budget
}

// estimateContextTokens estimates the transcript size. When an earlier assistant
// turn reported usage, that count is used for the prefix it covers and only the
// later messages are estimated.
func estimateContextTokens(messages []aimix.Message) int {
	lastUsage, usageTokens := -1, 0
	for i, m := range messages {
		am, ok := m.(aimix.AssistantMessage)
		if !ok || am.StopReason == aimix.StopReasonError || am.StopReason == aimix.StopReasonAborted {
			continue
		}
		if t := usageTokensOf(am.Usage); t > 0 {
			lastUsage, usageTokens = i, t
		}
	}
	tokens := usageTokens
	for _, m := range messages[lastUsage+1:] {
		tokens += estimateMessageTokens(m)
	}
	return tokens
}

func usageTokensOf(u aimix.Usage) int {
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.Input + u.Output + u.CacheRead + u.CacheWrite
}

func estimateMessageTokens(m aimix.Message) int {
	chars := 0
	switch v := m.(type) {
	case aimix.SystemMessage:
		chars = utf8.RuneCountInString(aimix.TextOf(v.Content))
		if len(v.ToolsAdded) > 0 {
			raw, _ := json.Marshal(v.ToolsAdded)
			chars += len(raw)
		}
	case aimix.UserMessage:
		chars = contentChars(v.Content)
	case aimix.ToolResultMessage:
		chars = contentChars(v.Content)
	case aimix.AssistantMessage:
		chars = contentChars(v.Content)
	}
	return int(math.Ceil(float64(chars) / charsPerToken))
}

func contentChars(cs aimix.Contents) int {
	chars := 0
	for _, c := range cs {
		switch b := c.(type) {
		case aimix.TextContent:
			chars += utf8.RuneCountInString(b.Text)
		case aimix.ThinkingContent:
			chars += utf8.RuneCountInString(b.Thinking)
		case aimix.ImageContent:
			chars += estimatedImageChars
		case aimix.ToolCallContent:
			chars += len(b.Name) + len(b.Arguments)
		}
	}
	return chars
}
