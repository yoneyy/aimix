package aimix

// Api identifies a wire protocol. Each Api has one implementation in the api package.
type Api string

func (a Api) String() string {
	return string(a)
}

const (
	ApiOpenAICompletions Api = "openai-completions"
	ApiOpenAIResponses   Api = "openai-responses"
	ApiAnthropicMessages Api = "anthropic-messages"
)
