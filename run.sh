#!/bin/bash

check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "You are not running as root. Commands requiring root will use 'sudo'."
        SUDO="sudo"
    else
        echo "You are running as root. 'sudo' is not required."
        SUDO=""
    fi
}

# Run the root check
check_root

# Initialize variables
validator_script="neurons/validators/validator_service.py"
autoRunLoc=$(readlink -f "$0")
api_proc_name="desearch_api_process"
validator_proc_name="desearch_validator_process"
args=()
version_location="./desearch/__init__.py"
version="__version__"

# Default values for API configuration
api_port="8005"
api_workers="4"

old_args=$@

# Check if pm2 is installed
if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

# Checks if $1 is smaller than $2
# If $1 is smaller than or equal to $2, then true. 
# else false.
version_less_than_or_equal() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

# Checks if $1 is smaller than $2
# If $1 is smaller than $2, then true. 
# else false.
version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

# Returns the difference between 
# two versions as a numerical value.
get_version_difference() {
    local tag1="$1"
    local tag2="$2"

    # Extract the version numbers from the tags
    local version1=$(echo "$tag1" | sed 's/v//')
    local version2=$(echo "$tag2" | sed 's/v//')

    # Split the version numbers into an array
    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    # Calculate the numerical difference
    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}

        # Compare the numbers and update the difference
        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

read_version_value() {
    # Read each line in the file
    while IFS= read -r line; do
        # Check if the line contains the variable name
        if [[ "$line" == *"$version"* ]]; then
            # Extract the value of the variable
            local value=$(echo "$line" | awk -F '=' '{print $2}' | tr -d ' ')
            strip_quotes $value
            return 0
        fi
    done < "$version_location"

    echo ""
}

check_package_installed() {
    local package_name="$1"
    os_name=$(uname -s)
    
    if [[ "$os_name" == "Linux" ]]; then
        # Use dpkg-query to check if the package is installed
        if dpkg-query -W -f='${Status}' "$package_name" 2>/dev/null | grep -q "installed"; then
            return 1
        else
            return 0
        fi
    elif [[ "$os_name" == "Darwin" ]]; then
         if brew list --formula | grep -q "^$package_name$"; then
            return 1
        else
            return 0
        fi
    else
        echo "Unknown operating system"
        return 0
    fi
}

check_variable_value_on_github() {
    local repo="$1"
    local file_path="$2"
    local variable_name="$3"

    local url="https://api.github.com/repos/$repo/contents/$file_path"
    local response=$(curl -s "$url")

    # Check if the response contains an error message
    if [[ $response =~ "message" ]]; then
        echo "Error: Failed to retrieve file contents from GitHub."
        return 1
    fi

    # Extract the content from the response
    local content=$(echo "$response" | tr -d '\n' | jq -r '.content')

    if [[ "$content" == "null" ]]; then
        echo "File '$file_path' not found in the repository."
        return 1
    fi

    # Decode the Base64-encoded content
    local decoded_content=$(echo "$content" | base64 --decode)

    # Extract the variable value from the content
    local variable_value=$(echo "$decoded_content" | grep "$variable_name" | awk -F '=' '{print $2}' | tr -d ' ')

    if [[ -z "$variable_value" ]]; then
        echo "Variable '$variable_name' not found in the file '$file_path'."
        return 1
    fi

    strip_quotes $variable_value
}

strip_quotes() {
    local input="$1"

    # Remove leading and trailing quotes using parameter expansion
    local stripped="${input#\"}"
    stripped="${stripped%\"}"

    echo "$stripped"
}

# Loop through all command line arguments
while [[ $# -gt 0 ]]; do
  arg="$1"

  # Check if the argument starts with a hyphen (flag)
  if [[ "$arg" == -* ]]; then
    # Check for standard param format 
    if [[ "$arg" == "--port" && $# -gt 1 ]]; then
      api_port="$2"
      shift 2
    elif [[ "$arg" == "--workers" && $# -gt 1 ]]; then
      api_workers="$2"
      shift 2
    # Check if the argument has a value
    elif [[ $# -gt 1 && "$2" != -* ]]; then
      # Add '=' sign between flag and value
      args+=("'$arg'");
      args+=("'$2'");
      shift 2
    else
      # Add '=True' for flags with no value
      args+=("'$arg'");
      shift
    fi
  else
    # Argument is not a flag, add it as it is
    args+=("'$arg '");
    shift
  fi
done

# Verify installation
if redis-cli --version; then
    echo "Redis already installed."
else
    $SUDO apt update
    # Install Redis
    $SUDO apt install -y redis
fi

echo "Attempting to start Redis using systemctl..."
if $SUDO systemctl start redis 2>/dev/null; then
    echo "Redis started successfully using systemctl."
else
    echo "systemctl not available or failed. Starting Redis manually..."
    if redis-server --daemonize yes; then
        echo "Redis started manually in the background."
    else
        echo "Failed to start Redis. Check your setup."
        exit 1
    fi
fi

branch=$(git branch --show-current)            # get current branch.
echo watching branch: $branch

# Get the current version locally.
current_version=$(read_version_value)

# Check if scripts are already running with pm2
if pm2 status | grep -q $api_proc_name; then
    echo "The API process is already running with pm2. Stopping and restarting..."
    pm2 delete $api_proc_name
fi

if pm2 status | grep -q $validator_proc_name; then
    echo "The validator process is already running with pm2. Stopping and restarting..."
    pm2 delete $validator_proc_name
fi

# Join the arguments with commas using printf
joined_args=$(printf "%s," "${args[@]}")

# Remove the trailing comma
joined_args=${joined_args%,}

# Create the pm2 config file with configurable port and workers
echo "module.exports = {
    apps: [
        {
            name: '$api_proc_name',
            script: 'uvicorn',
            interpreter: 'python3',
            args: [
                'neurons.validators.api:app',
                '--host',
                '0.0.0.0',
                '--port',
                '$api_port',
                '--workers',
                '$api_workers',
            ],
            exec_mode: 'fork',
        },
        {
            name: '$validator_proc_name',
            script: '$validator_script',
            interpreter: 'python3',
            min_uptime: '5m',
            max_restarts: '5',
            args: [$joined_args],
        },
    ],
}" > app.config.js

# Print configuration to be used
echo "Running with the following pm2 config:"
cat app.config.js
echo "API Configuration: Port=$api_port, Workers=$api_workers"

pm2 start app.config.js

# Check if packages are installed.
check_package_installed "jq"
if [ "$?" -eq 1 ]; then
    while true; do

        # First ensure that this is a git installation
        if [ -d "./.git" ]; then

            # check value on github remotely
            # Attempt to check the variable value on GitHub for both repositories
            latest_version=""
            repos=("Desearch-ai/subnet-22")

            for repo in "${repos[@]}"; do
                latest_version=$(check_variable_value_on_github "$repo" "desearch/__init__.py" "__version__")
                if [ $? -eq 0 ]; then
                    echo "Successfully retrieved version from $repo"
                    
                    # Set the working repo as the git remote origin URL
                    git remote set-url origin "https://github.com/$repo.git"
                    echo "Set git remote origin to https://github.com/$repo.git"
                    
                    break
                else
                    echo "Failed to retrieve version from $repo"
                fi
            done

            # If no version could be retrieved, exit with an error
            if [ -z "$latest_version" ]; then
                echo "Error: Could not retrieve version from any repository."
                exit 1
            fi

            # If the file has been updated
            if version_less_than $current_version $latest_version; then
                echo "latest version $latest_version"
                echo "current version $current_version"
                diff=$(get_version_difference $latest_version $current_version)
                if [ "$diff" -eq 1 ]; then
                    echo "current validator version:" "$current_version" 
                    echo "latest validator version:" "$latest_version" 

                    # Pull latest changes
                    # Failed git pull will return a non-zero output
                    if git pull origin $branch; then
                        # latest_version is newer than current_version, should download and reinstall.
                        echo "New version published. Updating the local copy."

                        # Install latest changes just in case.
                        pip install -e .

                        # Restart PM2 processes
                        echo "Restarting PM2 processes"
                        pm2 restart $api_proc_name
                        pm2 restart $validator_proc_name

                        # Update current version:
                        current_version=$(read_version_value)
                        echo ""

                        # Restart autorun script
                        echo "Restarting script..."
                        ./$(basename $0) $old_args && exit
                    else
                        echo "**Will not update**"
                        echo "It appears you have made changes on your local copy. Please stash your changes using git stash."
                    fi
                else
                    # current version is newer than the latest on git. This is likely a local copy, so do nothing. 
                    echo "**Will not update**"
                    echo "The local version is $diff versions behind. Please manually update to the latest version and re-run this script."
                fi
            else
                echo "**Skipping update **"
                echo "$current_version is the same as or more than $latest_version. You are likely running locally."
            fi
        else
            echo "The installation does not appear to be done through Git. Please install from source at https://github.com/opentensor/validators and rerun this script."
        fi
        
        # Wait about 30 minutes
        # This should be plenty of time for validators to catch up
        # and should prevent any rate limitations by GitHub.
        sleep 1200
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi