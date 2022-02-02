import os
 
# Function to rename multiple files
def main():

    folder = "15"
    for count, filename in enumerate(os.listdir(folder)):
        src =f"{folder}/{filename}"
        count = count+80
        name = f"img{str(count)}.jpg"
        # rename() function will
        # rename all the files
        os.rename(src, name)

if __name__ == '__main__':
    main()