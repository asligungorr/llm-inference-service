from celery_app import celery_app
from services import redis_client, clean_output, limit_sentences, HF_MODEL_ID, client

@celery_app.task
def run_async_inference(job_id: str, prompt: str, sentences: int):
    try:
        redis_client.set(f"job:{job_id}:status","running")

        system_prompt = (
            f"Answer in exactly {sentences} sentences. "
            "Do not include reasoning or explanations. "
        )

        completion = client.chat.completions.create(
            model= HF_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature= 0.4,
            max_tokens= 100
        )

        raw_output = completion.choices[0].message.content
        cleaned = clean_output(raw_output)
        final = limit_sentences(cleaned, sentences)

        redis_client.set(f"job:{job_id}:result", final)
        redis_client.set(f"job:{job_id}:status", "completed")

    except Exception as e:
        redis_client.set(f"job:{job_id}:status", "failed")
        redis_client.set(f"job:{job_id}:error", str(e))
        