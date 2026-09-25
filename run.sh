#!/bin/bash

validator_script="neurons/validators/validator.py"
api_proc_name="desearch_api_process"
validator_proc_name="desearch_validator_process"
args=()
version_location="./desearch/__init__.py"
version="__version__"

old_args=$@

if ! command -v pm2 &> /dev/null
then
    echo "pm2 could not be found. To install see: https://pm2.keymetrics.io/docs/usage/quick-start/"
    exit 1
fi

version_less_than_or_equal() {
    [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

version_less_than() {
    [ "$1" = "$2" ] && return 1 || version_less_than_or_equal $1 $2
}

get_version_difference() {
    local version1=$(echo "$1" | sed 's/v//')
    local version2=$(echo "$2" | sed 's/v//')

    IFS='.' read -ra version1_arr <<< "$version1"
    IFS='.' read -ra version2_arr <<< "$version2"

    local diff=0
    for i in "${!version1_arr[@]}"; do
        local num1=${version1_arr[$i]}
        local num2=${version2_arr[$i]}
        if (( num1 > num2 )); then
            diff=$((diff + num1 - num2))
        elif (( num1 < num2 )); then
            diff=$((diff + num2 - num1))
        fi
    done

    strip_quotes $diff
}

read_version_value() {
    while IFS= read -r line; do
        if [[ "$line" == *"$version"* ]]; then
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

    if [[ $response =~ "message" ]]; then
        echo "Error: Failed to retrieve file contents from GitHub."
        return 1
    fi

    local content=$(echo "$response" | tr -d '\n' | jq -r '.content')

    if [[ "$content" == "null" ]]; then
        echo "File '$file_path' not found in the repository."
        return 1
    fi

    local decoded_content=$(echo "$content" | base64 --decode)
    local variable_value=$(echo "$decoded_content" | grep "$variable_name" | awk -F '=' '{print $2}' | tr -d ' ')

    if [[ -z "$variable_value" ]]; then
        echo "Variable '$variable_name' not found in the file '$file_path'."
        return 1
    fi

    strip_quotes $variable_value
}

strip_quotes() {
    local input="$1"
    local stripped="${input#\"}"
    stripped="${stripped%\"}"

    echo "$stripped"
}

while [[ $# -gt 0 ]]; do
  arg="$1"

  if [[ "$arg" == -* ]]; then
    # Options of the removed API process, still passed by older launch commands.
    if [[ ("$arg" == "--port" || "$arg" == "--workers") && $# -gt 1 ]]; then
      shift 2
    elif [[ $# -gt 1 && "$2" != -* ]]; then
      args+=("'$arg'");
      args+=("'$2'");
      shift 2
    else
      args+=("'$arg'");
      shift
    fi
  else
    args+=("'$arg '");
    shift
  fi
done

branch=$(git branch --show-current)
echo watching branch: $branch

current_version=$(read_version_value)

if pm2 status | grep -q $api_proc_name; then
    echo "Removing the retired API process..."
    pm2 delete $api_proc_name
fi

if pm2 status | grep -q $validator_proc_name; then
    echo "The validator process is already running with pm2. Stopping and restarting..."
    pm2 delete $validator_proc_name
fi

joined_args=$(printf "%s," "${args[@]}")
joined_args=${joined_args%,}

echo "module.exports = {
    apps: [
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

echo "Running with the following pm2 config:"
cat app.config.js

pm2 start app.config.js

check_package_installed "jq"
if [ "$?" -eq 1 ]; then
    while true; do
        if [ -d "./.git" ]; then
            latest_version=""
            repos=("Desearch-ai/subnet-22")

            for repo in "${repos[@]}"; do
                latest_version=$(check_variable_value_on_github "$repo" "desearch/__init__.py" "__version__")
                if [ $? -eq 0 ]; then
                    echo "Successfully retrieved version from $repo"
                    git remote set-url origin "https://github.com/$repo.git"
                    echo "Set git remote origin to https://github.com/$repo.git"
                    break
                else
                    echo "Failed to retrieve version from $repo"
                fi
            done

            if [ -z "$latest_version" ]; then
                echo "Error: Could not retrieve version from any repository."
                exit 1
            fi

            if version_less_than $current_version $latest_version; then
                echo "latest version $latest_version"
                echo "current version $current_version"
                diff=$(get_version_difference $latest_version $current_version)
                if [ "$diff" -eq 1 ]; then
                    echo "current validator version:" "$current_version"
                    echo "latest validator version:" "$latest_version"

                    if git pull origin $branch; then
                        echo "New version published. Updating the local copy."

                        # scalecodec shadows cyscale's codec shim and breaks weight setting on bittensor 10.3.0.
                        pip uninstall async-substrate-interface substrate-interface scalecodec cyscale -y
                        pip install -e .

                        # The new run.sh recreates the validator process from its own config.
                        echo "Restarting script..."
                        ./$(basename $0) $old_args && exit
                    else
                        echo "**Will not update**"
                        echo "It appears you have made changes on your local copy. Please stash your changes using git stash."
                    fi
                else
                    echo "**Will not update**"
                    echo "The local version is $diff versions behind. Please manually update to the latest version and re-run this script."
                fi
            else
                echo "**Skipping update **"
                echo "$current_version is the same as or more than $latest_version. You are likely running locally."
            fi
        else
            echo "The installation does not appear to be done through Git. Please install from source at https://github.com/Desearch-ai/subnet-22 and rerun this script."
        fi

        # About 20 minutes, which keeps clear of GitHub's rate limits.
        sleep 1200
    done
else
    echo "Missing package 'jq'. Please install it for your system first."
fi
