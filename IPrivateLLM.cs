using OutSystems.ExternalLibraries.SDK;

namespace PrivateLLM
{
    [OSInterface(Name = "PrivateLLM", Description= "Secure, free, and fully local AI for ODC.", IconResourceName = "PrivateLLM.resources.privatellm_logo.png")]
    public interface IPrivateLLM
    {
        [OSAction (ReturnName = "Response")]
        public Response Call(
            string SystemPrompt,
            string UserPrompt,
            bool UseCustomParameters = false,
            CustomParameters? CustomParameters = null,
            [OSParameter(Description = "URL of the model file in the \"huggingface.co\" CDN. Default: \"https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_S.gguf\"")]
            string ModelFileURL = "",
            [OSParameter(Description = "Some models require a FREE Hugging Face access token to be fetched.")]
            string HFAccessToken = ""
        );

        [OSAction(ReturnName = "Responses")]
        public List<Response> BulkCall(
            List<Request> Requests,
            [OSParameter(Description = "URL of the model file in the \"huggingface.co\" CDN. Default: \"https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_S.gguf\"")]
            string ModelFileURL = "",
            [OSParameter(Description = "Some models require a FREE Hugging Face access token to be fetched.")]
            string HFAccessToken = ""
        );

        [OSAction(Description = "Dummy action to be used to keep the container alive, thus preventing \"cold-starts\".")]
        public void Ping();
    }
}