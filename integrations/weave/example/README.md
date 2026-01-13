# Accessing Weights & Biases Weave

You need to have a Weave account to use this feature. You can sign up for free at https://wandb.ai/site.

You then need to set the `WANDB_API_KEY:` environment variable with your Weights & Biases API key. You should find 
your API key in https://wandb.ai/home when you are logged in. 

You should then head to `https://wandb.ai/<user_name>/projects` and see the complete trace for your pipeline under
the pipeline name you specified, when creating the `WeaveConnector`.
