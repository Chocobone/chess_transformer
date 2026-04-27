import os
import shutil

def move_images(source_dir, target_dir):
    """
    source_dir 하위의 모든 png, jpg 파일을 찾아 target_dir로 이동합니다.
    파일명이 중복될 경우 자동으로 이름을 변경합니다.
    """
    # 이동 대상 확장자 리스트 (대소문자 구분 없이 처리하기 위해 소문자로 작성)
    extensions = {'.png', '.jpg', '.jpeg'}
    
    # 목적지 디렉토리가 없으면 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"📁 폴더 생성 완료: {target_dir}")

    count = 0

    # os.walk를 사용하여 하위 디렉토리까지 모두 탐색
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 파일 확장자 확인 (소문자로 변환하여 비교)
            _, ext = os.path.splitext(file)
            if ext.lower() in extensions:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(target_dir, file)

                # 중복 파일명 처리 (덮어쓰기 방지)
                base_name = file
                duplicate_count = 1
                while os.path.exists(dst_path):
                    name, extension = os.path.splitext(base_name)
                    new_filename = f"{name}_{duplicate_count}{extension}"
                    dst_path = os.path.join(target_dir, new_filename)
                    duplicate_count += 1

                # 파일 이동
                try:
                    shutil.move(src_path, dst_path)
                    print(f"✅ 이동 완료: {file} -> {dst_path}")
                    count += 1
                except Exception as e:
                    print(f"❌ 오류 발생 ({file}): {e}")

    print(f"\n총 {count}개의 파일을 이동했습니다.")

# --- 사용 설정 ---
# 아래 경로를 실제 경로로 수정해서 사용하세요.
# Windows 예시: r"C:\Users\Name\Downloads\Source"
# Mac/Linux 예시: "/Users/Name/Downloads/Source"

SOURCE_DIRECTORY = r"C:\Users\GazziLabs_\Music\obsidian_PARA" 
DESTINATION_DIRECTORY = r"C:\Users\GazziLabs_\Music\obsidian_PARA\0. BASE\Photos"

# 함수 실행
if __name__ == "__main__":
    move_images(SOURCE_DIRECTORY, DESTINATION_DIRECTORY)