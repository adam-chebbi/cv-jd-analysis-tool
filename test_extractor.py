from src.extractor import SkillExtractor

def main():
    extractor = SkillExtractor()
    skills = extractor.process_cv("uploads/cv_Adam_Chebbi.pdf")
    print(f"Extracted skills: {skills}")

if __name__ == "__main__":
    main()
