# Text Generation
---

## Project Setup

Set Python 3.11.6 as the Local Version
 - Navigate to the project directory and set the local version:

```
pyenv local 3.11.6
```
### Creating a Virtual Environment

Create the Virtual Environment
- Use virtualenv to create an isolated environment with Python 3.11.6:

```
virtualenv venv --python=$(pyenv which python)
```

Activate the Virtual Environment

```
source venv/bin/activate
```

### Installing Required Libraries

Install the Python libraries:

```
pip install -r requirements.txt
```

### Running the Project

```
python hello_world.py
```

You should see output like

```
[{'label': 'POSITIVE', 'score': 0.999...}]
```

Alright, time to add some more interesting scripts to our tutorial!

---