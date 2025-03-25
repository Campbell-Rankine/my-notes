import requests
from uuid import uuid4
import asyncio
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait
from time import sleep
from timeit import default_timer as timer

workers = multiprocessing.cpu_count()
BASE_URL = f"http://fastapi.localhost"


def request_server(uuid: str):
    x1: float = random.random()
    x2: float = random.random()
    print(x1, x2)

    request_obj = {
        "uuid": uuid,
        "x1": x1,
        "x2": x2,
    }
    response = requests.request(method="GET", url=f"{BASE_URL}/add", json=request_obj)
    print(response.status_code)
    print(response.json())
    return response


if __name__ == "__main__":
    # timer block
    start = timer()
    N = 10

    # create request job arguments
    args = [str(uuid4()) for _ in range(N)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(request_server, args)

    # resolve request results
    results = list(futures)
    print(results)
    assert len(results) == N

    end = timer()
    print(f"Parallel Jobs completed in {round(end-start, 4)}s")
    start = timer()
    for i in range(N):
        response = request_server(str(uuid4()))
        print("request made")
    end = timer()
    print(f"Serial Jobs completed in {round(end-start, 4)}s")
