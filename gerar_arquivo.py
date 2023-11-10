import subprocess

class CriadorExecutavel:
    def __init__(self, nome_arquivo_principal):
        self.arquivo_principal = nome_arquivo_principal

    def criar_executavel(self):
        comando = f"pyinstaller --onefile -n setup {self.arquivo_principal} --distpath ."
        processo = subprocess.Popen(comando, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = processo.communicate()

        if error:
            print(f"Ocorreu um erro: {error.decode('utf-8')}")
        else:
            print("O executável foi criado com sucesso.")

# Use a classe CriadorExecutavel
nome_arquivo = "setup.py"  # Substitua pelo nome do seu arquivo principal
criador = CriadorExecutavel(nome_arquivo_principal=nome_arquivo)
criador.criar_executavel()