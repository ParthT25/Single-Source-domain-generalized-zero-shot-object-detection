import os

def process_file(file_path, line_ranges, action):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        for start, end in line_ranges:
            for i in range(start - 1, end):
                stripped_line = lines[i].lstrip()
                leading_spaces = len(lines[i]) - len(stripped_line)
                
                if action == "uncomment" and stripped_line.startswith('# '):
                    lines[i] = ' ' * leading_spaces + stripped_line[2:]  # Remove '# '
                elif action == "uncomment" and stripped_line.startswith('#'):
                    lines[i] = ' ' * leading_spaces + stripped_line[1:]  # Remove '#'
                elif action == "comment" and not stripped_line.startswith('#'):
                    lines[i] = ' ' * leading_spaces + '# ' + stripped_line  # Add '# '

        with open(file_path, 'w') as file:
            file.writelines(lines)

    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except IndexError:
        print(f"Line number out of range in file {file_path}.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

def mode_1():
    process_file('/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/modeling/backbone.py', [(48, 50)], "uncomment")
    process_file('/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/modeling/meta_arch.py', [(238, 263), (539, 543)], "uncomment")

def mode_2():
    process_file('/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/modeling/backbone.py', [(48, 50)], "comment")
    process_file('/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/modeling/meta_arch.py', [(238, 263), (539, 543)], "comment")

def main():
    print("Mode 1: Setup for extraction of ROI feature maps")
    print("Mode 2: Setup for testing the model and disable ROI feature extraction")
    print("Note  : please disable extraction(Run mode 2) before testing the model")
    mode = input("Enter mode (1 or 2): ").strip()
    if mode == "1":
        mode_1()
        print("Mode 1: Code uncommented.")
    elif mode == "2":
        mode_2()
        print("Mode 2: Code commented.")
    else:
        print("Invalid mode. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
