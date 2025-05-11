import sys

def tee(files, mode='w'):
    """
    Read from stdin and write to stdout and files.
    
    Args:
        files: List of file names to write to
        mode: File opening mode ('w' for write, 'a' for append)
    """
    # Open all files
    file_handles = [open(filename, mode) for filename in files]
    
    try:
        # Read from stdin line by line
        for line in sys.stdin:
            # Write to stdout
            sys.stdout.write(line)
            
            # Write to all files
            for file in file_handles:
                file.write(line)
                file.flush()  # Ensure data is written immediately
                
    finally:
        # Close all files
        for file in file_handles:
            file.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Python implementation of tee command')
    parser.add_argument('files', nargs='*', help='Files to write to')
    parser.add_argument('-a', '--append', action='store_true', help='Append to files instead of overwriting')
    
    args = parser.parse_args()
    mode = 'a' if args.append else 'w'
    
    tee(args.files, mode)