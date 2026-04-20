@echo off
echo ========================================
echo Uploading SCOPE-X to GitHub
echo ========================================
echo.

cd /d "%~dp0"

echo Step 1: Initializing git repository...
git init
if errorlevel 1 (
    echo ERROR: git init failed. Please install Git for Windows.
    pause
    exit /b 1
)

echo Step 2: Adding files...
git add .

echo Step 3: Creating initial commit...
git commit -m "Initial commit: SCOPE-X V3.5.0 with Group Compression

- Dual CLS tokens for contrastive learning
- Multi-task decoder (value, cluster, chr, modality)
- Optional group compression in RAW feature space
- Fixed peak status prediction using ALL ATAC tokens
- DDP multi-GPU training support
- 13 experimental scripts included"

echo Step 4: Setting main branch...
git branch -M main

echo Step 5: Adding remote repository...
git remote add origin https://github.com/caiquanyou/SCOPE-X.git

echo Step 6: Pushing to GitHub...
git push -u origin main

echo.
echo ========================================
echo Upload complete!
echo ========================================
pause
