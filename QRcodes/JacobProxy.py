


FIFO = 'fromAidan'
terminating = False
while True:
    with open(FIFO) as fifo:
        while True:
            data = fifo.read().strip().strip('\x00')

            # i think this resets the writer as to not block
            if len(data) == 0:
                break

            # Check for termination code
            if data.startswith(('done', 'exit', 'close')):
                terminating = True

            # print what was received on the pipe
            print('------------------------------------------------')
            print(data)



        if terminating:
            fifo.close()
            break
    if terminating:
        break