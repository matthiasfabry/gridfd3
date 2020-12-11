def run_join_threads(threads):
    """
    runs and joins the threads passed in the list 'threads'. Also catches any exceptions thrown in the threads
    :param threads: list of threads to be run and joined
    """
    for thread in threads:
        try:
            thread.start()
        except Exception as e:
            print(repr(thread), e)

    for thread in threads:
        thread.join()
