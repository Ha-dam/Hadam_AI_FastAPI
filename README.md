# Hadam_AI_FastAPI
Hadam diary generartive server 

## Overview
This FastAPI server is designed to create and manage diary entries with the integration of OpenAI's GPT-3 for text generation and Stable Diffusion for image generation. For better output, we used DeepL API to translate tentative results. It provides endpoints to create new diary entries, regenerate diary text, and regenerate diary images based on user inputs.
Currently, 'Hadam' is running live on Amazon EC2, and the generated picture images are stored in an Amazon S3 bucket.

## Features
- **Diary Creation**:  Generate diary entries using GPT-3 based on user-provided keywords about their experiences, emotions, and details of the day.
- **Diary Regeneration**: Allows users to regenerate the text of existing diary entries if they are not satisfied with the current content.
- **Image Regeneration:**: Allows users to regenerate images for diary entries using Stable Diffusion API, if they are not satisfied with the current content.

## Installation 
To set up the server locally, follow these steps:

1. ### Clone the Repository
```
git clone [repository URL] 
cd [repository directory] 
```

2. ### Install Requirements
```pip install -r requirements.txt ```


3. ### Set Up Environment Variables
Create a .env file in the root directory with the following content:

```
DB_USER=[your_database_username]
DB_PASS=[your_database_password]
DB_HOST=[your_database_host]
DB_NAME=[your_database_name]
OPENAI_API_KEY=[your_openai_api_key]
STABLE_DIFFUSION_API_KEY=[your_stable_diffusion_api_key]
DEEPL_API_KEY=[your_deepl_api_key]
```

Note: Replace [repository URL], [your_database_username], [your_database_password], [your_database_host], [your_database_name], [your_openai_api_key], [your_stable_diffusion_api_key] and [your_deepl_api_key] with the actual values for your project. Adjust the content as needed based on your project's specific requirements and setup.

4. ### Run the Server
```uvicorn main:app --reload ```

## Usage
After starting the server, the following endpoints are available:

- **'POST /diary/create'**: Create a new diary entry.
- **'PUT /diary/recreate/{diary_id}'**: Regenerate the image of an existing diary entry.
- **'PUT /diary/regenerate-image/{diary_id}'**: Regenerate the image of an existing diary entry.

You can access these endpoints through the auto-generated FastAPI documentation at http://13.209.87.157:8000/docs

## Language and Tools

<div align=left>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon%20EC2-FF9900?style=for-the-badge&logo=Amazon%20EC2&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon%20S3-569A31?style=for-the-badge&logo=Amazon%20S3&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapia&logoColor=white">
  <img src="https://img.shields.io/badge/GPT3-412991?style=for-the-badge&logo=openai&logoColor=white">
  <img src="https://img.shields.io/badge/Deepl-0F2B46?style=for-the-badge&logo=deepl&logoColor=white">
  
</div>

## Author



