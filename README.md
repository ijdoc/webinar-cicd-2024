# wandb-webinar-cicd-2024
Discover how improving your CI/CD pipeline can boost your team’s productivity and ensure that your production models are always your best models.

<p align="center"><a href="http://www.youtube.com/watch?feature=player_embedded&v=Sw4M-b_GQZg
" target="_blank"><img src="http://img.youtube.com/vi/Sw4M-b_GQZg/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" border="10" /></a></p>

## Workflow Entry-Point

### Setup

#### Prepare the environment

We use pipenv to maintain python virtual environments across the project:
```bash
pip install pipenv
cd batch
pipenv sync
```

#### Create `.env` file

Within the `batch` folder, create a `.env` file. The variables defined here will be automatically loaded by `pipenv` into the python virtual environment at runtime.
```
# batch/.env
GITHUB_TOKEN="paste_your_token_here"
```

### Trigger the workload
```bash
cd batch
pipenv run python trigger_batch_data.py --repo ijdoc/webinar-cicd-2024 --iteration=25
```
