import os
import sys
import subprocess
from pathlib import Path

def get_runpath(library_path):
    """
    Get RUNPATH from a shared library using readelf
    
    Args:
        library_path (str or Path): Path to the shared library
        
    Returns:
        list: List of paths from RUNPATH
    """
    try:
        # readelf 실행
        result = subprocess.run(
            ['readelf', '-d', str(library_path)], 
            capture_output=True, 
            text=True, 
            check=True
        )

        # 출력에서 RUNPATH 찾기
        for line in result.stdout.split('\n'):
            if 'RUNPATH' in line:
                # RUNPATH 문자열 파싱
                paths = line.split('[')[-1].strip(']').split(':')
                
                # $ORIGIN 처리
                lib_dir = os.path.dirname(os.path.abspath(library_path))
                processed_paths = []
                
                for path in paths:
                    if path.startswith('$ORIGIN'):
                        # $ORIGIN을 라이브러리 위치로 대체
                        path = path.replace('$ORIGIN', lib_dir)
                    processed_paths.append(path)
                
                return processed_paths
        return []
        
    except subprocess.CalledProcessError:
        print(f"Error: Could not read RUNPATH from {library_path}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def add_library_paths(library_path):
    """
    Add library's RUNPATH to Python's sys.path
    
    Args:
        library_path (str or Path): Path to the shared library
    """
    library_path = Path(library_path)
    if not library_path.exists():
        print(f"Error: Library {library_path} does not exist")
        return
    
    # RUNPATH 가져오기
    paths = get_runpath(library_path)
    
    # 기존 sys.path에 없는 경로만 추가
    for path in paths:
        if path and os.path.exists(path) and path not in sys.path:
            sys.path.append(path)
            # LD_LIBRARY_PATH도 업데이트
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if path not in current_ld_path:
                if current_ld_path:
                    os.environ['LD_LIBRARY_PATH'] = f"{path}:{current_ld_path}"
                else:
                    os.environ['LD_LIBRARY_PATH'] = path