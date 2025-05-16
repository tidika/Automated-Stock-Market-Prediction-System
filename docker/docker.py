import boto3
import base64
import subprocess
from botocore.exceptions import ClientError
from config import AWS_REGION, ACCOUNT_ID, REPOSITORY_NAME, IMAGE_TAG

# === Configuration ===
aws_region = AWS_REGION
account_id = ACCOUNT_ID
repository_name = REPOSITORY_NAME
image_tag = IMAGE_TAG


def login_to_ecr(region: str, registry_id: str) -> None:
    """
    Logs in to Amazon ECR using Docker from within a Python script.

    :param region: AWS region (e.g., 'us-east-2').
    :param registry_id: AWS account ID (e.g., '123456789098').
    """
    try:
        ecr_client = boto3.client("ecr", region_name=region)
        response = ecr_client.get_authorization_token(registryIds=[registry_id])
        auth_data = response["authorizationData"][0]

        # Decode authorization token
        token = base64.b64decode(auth_data["authorizationToken"]).decode()
        username, password = token.split(":")

        registry_url = auth_data["proxyEndpoint"]

        # Run docker login command
        cmd = [
            "docker",
            "login",
            "--username",
            username,
            "--password-stdin",
            registry_url,
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        process.communicate(input=password.encode())

        if process.returncode == 0:
            print(f"✅ Successfully logged in to ECR: {registry_url}")
        else:
            raise Exception("Docker login failed.")

    except ClientError as e:
        print(f"AWS Error: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"Login failed: {str(e)}")


# def build_docker_image(repository: str, tag: str) -> None:
#     """Builds the Docker image locally."""
#     build_command = f"docker build -t {repository}:{tag} ."
#     try:
#         subprocess.run(
#             build_command, check=True, capture_output=True, text=True, encoding="utf-8"
#         )
#         print("✅ Docker image built successfully.")
#     except subprocess.CalledProcessError as e:
#         print("Docker build failed!\n", e.stderr)
#         exit(1)

def build_docker_image(repository: str, tag: str) -> None:
    """Builds the Docker image locally or in GitHub Actions."""
    build_command = ["docker", "build", "-t", f"{repository}:{tag}", "."]

    try:
        print(f"🔧 Running Docker build: {' '.join(build_command)}")
        result = subprocess.run(
            build_command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8"

        )
        print("✅ Docker image built successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Docker build failed!")
        print(e.stderr)
        exit(1)

# def tag_docker_image(local_repo: str, tag: str, full_image_name: str) -> None:
#     """Tags the Docker image with the ECR repo URL."""
#     tag_command = f"docker tag {local_repo}:{tag} {full_image_name}"
#     subprocess.run(tag_command, check=True)
#     print(f"✅ Docker image tagged as {full_image_name}")

def tag_docker_image(local_repo: str, tag: str, full_image_name: str) -> None:
    """Tags the Docker image with the ECR repo URL using a safe, structured command."""
    tag_command = ["docker", "tag", f"{local_repo}:{tag}", full_image_name]
    subprocess.run(tag_command, check=True)
    print(f"✅ Docker image tagged as {full_image_name}")


# def push_docker_image(full_image_name: str) -> None:
#     """Pushes the Docker image to ECR."""
#     push_command = f"docker push {full_image_name}"
#     try:
#         subprocess.run(push_command, check=True, capture_output=True, text=True)
#         print(f"✅ Docker image pushed to {full_image_name}")
#     except subprocess.CalledProcessError as e:
#         print("Docker push failed!\n", e.stderr)
#         exit(1)

def push_docker_image(full_image_name: str) -> None:
    """Pushes the Docker image to ECR using a safe subprocess command."""
    push_command = ["docker", "push", full_image_name]
    try:
        result = subprocess.run(push_command, check=True, capture_output=True, text=True)
        print(f"✅ Docker image pushed to {full_image_name}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Docker push failed!")
        print(e.stderr)
        exit(1)


def main():
    """Main function to execute the Docker operations."""
    # Step 1: Login to ECR
    login_to_ecr(region=aws_region, registry_id=account_id)

    # Step 2: Define full image name
    full_image_name = (
        f"{account_id}.dkr.ecr.{aws_region}.amazonaws.com/{repository_name}:{image_tag}"
    )

    # Step 3: Build Docker image
    build_docker_image(repository_name, image_tag)

    # Step 4: Tag image
    tag_docker_image(repository_name, image_tag, full_image_name)

    # Step 5: Push image
    push_docker_image(full_image_name)


if __name__ == "__main__":
    main()
