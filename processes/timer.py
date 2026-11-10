import time
import multiprocessing as mp

def timer(frequency,event : mp.Event):

    print(f"Timer started at {frequency} Hz")
    delta = 1/frequency
    next_call = time.perf_counter() + delta

    while True :
        now = time.perf_counter()
        sleep_time = next_call - now

        if sleep_time > 0:
            time.sleep(sleep_time)

        event.set()

        # On calcule le prochain target de manière absolue pour éviter la dérive
        next_call += delta