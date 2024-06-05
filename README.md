# 🌟 Hadam_AI_FastAPI
**Hadam Diary Generative Server**

## Overview
Welcome to the Hadam Diary Generative Server! This FastAPI server enables the creation and management of diary entries, integrating cutting-edge technologies:
- **OpenAI's GPT-3** for text generation.
- **Stable Diffusion** for image generation.
- **DeepL API** for translating tentative results.

Hadam is live on Amazon EC2, with generated images stored in an Amazon S3 bucket.

## Features
- **📔 Diary Creation**: Generate diary entries using GPT-3 based on user-provided keywords about their experiences, emotions, and details of the day.
- **✍️ Diary Regeneration**: Allows users to regenerate the text of existing diary entries if they are not satisfied with the current content.
- **🖼️ Image Regeneration**: Allows users to regenerate images for diary entries using the Stable Diffusion API if they are not satisfied with the current content.

## Installation 
To set up the server locally, follow these steps:

1. **Clone the Repository**
    ```sh
    git clone [repository URL]
    cd [repository directory]
    ```

2. **Install Requirements**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**
    Create a `.env` file in the root directory with the following content:
    ```env
    DB_USER=[your_database_username]
    DB_PASS=[your_database_password]
    DB_HOST=[your_database_host]
    DB_NAME=[your_database_name]
    OPENAI_API_KEY=[your_openai_api_key]
    STABLE_DIFFUSION_API_KEY=[your_stable_diffusion_api_key]
    DEEPL_API_KEY=[your_deepl_api_key]
    ```

    *Note: Replace placeholders with actual values.*

4. **Run the Server**
    ```sh
    uvicorn main:app --reload
    ```

## Usage
After starting the server, the following endpoints are available:

- **POST `/diary/create`**: Create a new diary entry.
- **PUT `/diary/recreate/{diary_id}`**: Regenerate the text of an existing diary entry.
- **PUT `/diary/regenerate-image/{diary_id}`**: Regenerate the image of an existing diary entry.

Access the auto-generated FastAPI documentation [here](http://13.209.87.157:8000/docs).

## Language and Tools
<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon%20EC2-FF9900?style=for-the-badge&logo=Amazon%20EC2&logoColor=white">
  <img src="https://img.shields.io/badge/Amazon%20S3-569A31?style=for-the-badge&logo=Amazon%20S3&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/GPT3-412991?style=for-the-badge&logo=openai&logoColor=white">
  <img src="https://img.shields.io/badge/DeepL-0F2B46?style=for-the-badge&logo=deepl&logoColor=white">
</div>

## Author
*Your Name*




