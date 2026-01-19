from celery_app import celery_app
from services import redis_client, clean_output, limit_sentences, HF_MODEL_ID, client
from prometheus_client import Histogram, Counter 



TASK_LATENCY = Histogram("celery_task_duration_seconds", "Task duration", ["task"])
TASK_FAILURES = Counter("celery_task_failures_total", "Task failures", ["task"])

@celery_app.task(
        bind=True,
        autoretry_for=(Exception,),
        retry_backoff=True,
        retry_backoff_max=60,
        retry_jitter=True,
        retry_kwargs={"max_retries": 5},
)

def run_async_inference(self, job_id: str, prompt: str, sentences: int):
    with TASK_LATENCY.labels(task="run_async_inference").time():
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
            TASK_FAILURES.labels(task="run_async_inference").inc()
            redis_client.set(f"job:{job_id}:status", "failed")
            redis_client.set(f"job:{job_id}:error", str(e))
        