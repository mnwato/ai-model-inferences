from diffusers import StableDiffusionPipeline
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
import uvicorn
from time import time


app = FastAPI()


model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda"
imgs_path = "./imgs/"

pipeline = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
    )
pipeline = pipeline.to(device)



@app.get("/")
async def welcome_page():
    return "Life is easier with AI :)"


@app.get("/stableDiffusion")
async def stable_diffusion(prompt, request:Request):
    if prompt and type(prompt)==str:
        try:
            cur_time = int(time())
            print(f"prompt: {prompt} at {str(cur_time)}")
            image = pipeline(prompt).images[0]
            filename = imgs_path + f"{cur_time}.png"
            image.save(filename)
            return FileResponse(filename)
        except Exception as e:
            print(f"prompt: {prompt} at {str(cur_time)} got error: {str(e)}")
            raise HTTPException(
                status_code = 500)
    else:
        return "Enter a prompt"



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
