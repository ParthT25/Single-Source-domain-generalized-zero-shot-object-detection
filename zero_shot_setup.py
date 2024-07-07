import yaml

def update_config_file(config_path, num_new_classes):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Update NUM_CLASSES
        if 'MODEL' in config and 'ROI_HEADS' in config['MODEL'] and 'NUM_CLASSES' in config['MODEL']['ROI_HEADS']:
            config['MODEL']['ROI_HEADS']['NUM_CLASSES'] += num_new_classes

        with open(config_path, 'w') as file:
            yaml.safe_dump(config, file)

    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
    except Exception as e:
        print(f"An error occurred while processing {config_path}: {e}")

def update_dataset_file(dataset_path, new_class_names):
    try:
        with open(dataset_path, 'r') as file:
            lines = file.readlines()

        # Find and check the all_class_name array
        for i in range(len(lines)):
            if 'all_class_name' in lines[i]:
                start_idx = i
                all_class_names_line = lines[i]
                while ']' not in lines[start_idx]:
                    start_idx += 1
                    all_class_names_line += lines[start_idx]
                all_class_names = eval(all_class_names_line.split('=')[1].strip())
                new_class_names_lower = [name.lower() for name in new_class_names]
                existing_class_names_lower = [name.lower() for name in all_class_names]

                if any(name in existing_class_names_lower for name in new_class_names_lower):
                    print("One or more new class names already exist in all_class_name. No updates made.")
                    return False

                lines[start_idx] = lines[start_idx].rstrip(']\n') + ', ' + ', '.join(f'"{name}"' for name in new_class_names) + ']\n'
                break

        with open(dataset_path, 'w') as file:
            file.writelines(lines)
        return True

    except FileNotFoundError:
        print(f"Dataset file {dataset_path} not found.")
        return False
    except Exception as e:
        print(f"An error occurred while processing {dataset_path}: {e}")
        return False

def main():
    config_path = '/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/configs/diverse_weather.yaml'
    dataset_path = '/u/student/2022/cs22mtech14005/Single-Source-domain-generalized-zero-shot-object-detection/data/datasets/diverse_weather.py'
    
    new_class_names = input("Enter the names of the new classes, separated by commas: ").strip().lower().split(',')
    num_new_classes = len(new_class_names)
    
    if update_dataset_file(dataset_path, new_class_names):
        update_config_file(config_path, num_new_classes)
        print("Config and dataset files updated successfully.")
    else:
        print("Config file not updated due to existing class names in the dataset.")

if __name__ == "__main__":
    main()
