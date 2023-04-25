# Use a PyTorch base image
FROM pytorch/pytorch:latest

# Set the working directory to /app
WORKDIR /app
# WORKDIR /app
COPY . /app
# Copy the code files to the container
# COPY opacus_test.py /app
# COPY requirements.txt /app
# Copy the dataset to the container
# COPY cifar10/ /app/cifar10/

# Install any additional dependencies
RUN pip install opacus tqdm
RUN pip install -r requirements.txt
# Define the command to run the code
CMD ["python3", "opacus_test.py"]