import importlib.metadata

def generate_requirements():
    installed_packages = {
        dist.metadata['Name']: dist.version 
        for dist in importlib.metadata.distributions()
    }
    with open("requirements.txt", "w", encoding="utf-8") as file:
        for package, version in installed_packages.items():
            file.write(f"{package}=={version}\n")
    print("✅ 최신 방식으로 requirements.txt 생성 완료!")

generate_requirements()
