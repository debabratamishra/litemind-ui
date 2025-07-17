from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ..models.user import User
from .. import get_db

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get('/signup', response_class=HTMLResponse)
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@router.post('/signup', response_class=RedirectResponse)
async def signup(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(User).filter_by(username=username))
        if result.scalars().first():
            request.session['flash'] = 'Username already exists.'
            return RedirectResponse(url='/auth/signup', status_code=303)
        role = 'admin' if not (await db.execute(select(User))).scalars().first() else 'user'
        user = User(username, password, role=role)
        db.add(user)
        await db.commit()
    request.session['flash'] = 'Signup successful! Please sign in.'
    return RedirectResponse(url='/auth/signin', status_code=303)

@router.get('/signin', response_class=HTMLResponse)
async def signin_page(request: Request):
    flash_message = request.session.pop('flash', None)
    return templates.TemplateResponse("signin.html", {"request": request, "flash_message": flash_message})

@router.post('/signin', response_class=RedirectResponse)
async def signin(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    async with db.begin():
        result = await db.execute(select(User).filter_by(username=username))
        user = result.scalars().first()
        if user and user.check_password(password):
            request.session['username'] = username
            request.session['role'] = user.role
            return RedirectResponse(url='/chat', status_code=303)
        request.session['flash'] = 'Invalid credentials.'
        return RedirectResponse(url='/auth/signin', status_code=303)

@router.get('/logout', response_class=RedirectResponse)
async def logout(request: Request):
    request.session.pop('username', None)
    request.session.pop('role', None)
    return RedirectResponse(url='/auth/signin', status_code=303)

@router.get('/users', response_class=HTMLResponse)
async def list_users(request: Request, db: AsyncSession = Depends(get_db)):
    if 'username' not in request.session or request.session.get('role') != 'admin':
        return RedirectResponse(url='/auth/signin', status_code=303)
    async with db.begin():
        result = await db.execute(select(User.username))
        users = result.scalars().all()
    return templates.TemplateResponse("users.html", {"request": request, "users": users})