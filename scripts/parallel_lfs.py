"""Use independent file handles for bounded parallel HTTP multipart uploads."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import huggingface_hub.lfs as lfs

_ORIGINAL=lfs._upload_parts_iteratively


def upload_parts_parallel(operation, sorted_parts_urls, chunk_size, workers=32):
    # Buffered file objects have a shared seek pointer; retain upstream behavior for them.
    if not isinstance(operation.path_or_fileobj,(str,Path)):
        return _ORIGINAL(operation,sorted_parts_urls,chunk_size)
    n_parts=len(sorted_parts_urls)
    size=operation.upload_info.size
    completed=0
    sent=0
    lock=threading.Lock()
    print(f'HTTP_MULTIPART_START {operation.path_in_repo} parts={n_parts} chunk_bytes={chunk_size} workers={min(workers,n_parts)}',flush=True)

    def upload_one(item):
        nonlocal completed,sent
        index,url=item
        # Each worker must have its own file descriptor: concurrent seeks on one
        # shared descriptor can silently send incorrect checkpoint bytes.
        with operation.as_file(with_tqdm=False) as fileobj:
            with lfs.SliceFileObj(fileobj,seek_from=chunk_size*index,read_limit=chunk_size) as body:
                response=lfs.http_backoff('PUT',url,data=body)
                lfs.hf_raise_for_status(response)
                headers=dict(response.headers)
        with lock:
            completed+=1
            sent+=min(chunk_size,size-chunk_size*index)
            if completed==1 or completed%10==0 or completed==n_parts:
                print(f'HTTP_MULTIPART_PROGRESS {operation.path_in_repo} parts={completed}/{n_parts} bytes={sent}/{size}',flush=True)
        return headers

    # map returns headers in part-number order even if network requests finish out of order.
    with ThreadPoolExecutor(max_workers=min(workers,max(1,n_parts))) as pool:
        result=list(pool.map(upload_one,enumerate(sorted_parts_urls)))
    print(f'HTTP_MULTIPART_COMPLETE {operation.path_in_repo}',flush=True)
    return result


def install():
    lfs._upload_parts_iteratively=upload_parts_parallel
