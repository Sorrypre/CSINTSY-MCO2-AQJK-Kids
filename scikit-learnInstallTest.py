'''
Run this program to check if you have successfully installed the scikit-learn library
'''
def main(): 
    print("Testing scikit-learn installation...\n")
    try:
        import sklearn
        print("scikit-learn is installed.")
        print(f"Version: {sklearn.__version__}")
    except ImportError:
        print("scikit-learn is not installed.")

if __name__ == "__main__":
    main()