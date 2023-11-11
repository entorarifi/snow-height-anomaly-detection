import asyncio
import os
import secrets

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from src.active_learner import ActiveLearner

load_dotenv()
LOG_FILE_PATH = os.getenv('AL_LOG_FILE_PATH')
API_KEY = os.getenv('AL_API_KEY')
BASIC_AUTH_USERNAME = os.getenv('AL_BASIC_AUTH_USERNAME')
BASIC_AUTH_PASSWORD = os.getenv('AL_BASIC_AUTH_PASSWORD')

app = FastAPI()
templates = Jinja2Templates(directory='templates')
security = HTTPBasic()


def validate_api_key(api_key: str):
    return api_key == API_KEY


def validate_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, BASIC_AUTH_USERNAME)
    correct_password = secrets.compare_digest(credentials.password, BASIC_AUTH_PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Incorrect email or password',
            headers={'WWW-Authenticate': 'Basic'},
        )
    return credentials


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket, api_key: str = None):
    if not validate_api_key(api_key):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()

    while True:
        if not os.path.exists(LOG_FILE_PATH) or os.path.getsize(LOG_FILE_PATH) == 0:
            await websocket.send_text('Log file is emtpy')
        else:
            with open(LOG_FILE_PATH, 'r') as file:
                log_content = file.read()
                await websocket.send_text(log_content)
        await asyncio.sleep(2)


@app.get('/', response_class=HTMLResponse)
async def get(request: Request, credentials: HTTPBasicCredentials = Depends(validate_credentials)):
    return templates.TemplateResponse('index.html', {'request': request, 'api_key': API_KEY})


@app.get('/iteration/start')
async def get(background_tasks: BackgroundTasks, credentials: HTTPBasicCredentials = Depends(validate_credentials)):
    active_learner = ActiveLearner.build()

    if active_learner.iteration.is_locked():
        return {'success': False, 'message': 'An iteration is already running. Please try again later.'}

    background_tasks.add_task(active_learner.run_iteration)
    return {'success': 'true', 'message': 'Iteration has been successfully initiated.'}


@app.get('/iteration/reset')
async def get(credentials: HTTPBasicCredentials = Depends(validate_credentials)):
    active_learner = ActiveLearner.build()
    active_learner.reset()
    return {'success': 'true', 'message': 'Environment has been successfully reset.'}
