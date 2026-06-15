_printed: set = set()


def log_once(message, accelerator=None):
    global _printed
    key = str(message)
    if key in _printed:
        return
    _printed.add(key)
    if accelerator is None or accelerator.is_main_process:
        print(message, flush=True)
