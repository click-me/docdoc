def find_all(str, substr):
    start = 0
    while True:
        start = str.find(substr, start)
        if start == -1: return
        yield start
        start += len(substr)  # use start += 1 to find overlapping matches
