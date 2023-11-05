from label_studio_sdk import Client, Project

class LabelStudioClient:
    def __init__(
        self,
        base_url,
        api_key,
        project_name,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.project_name = project_name
        self.client: Client = self.get_client()
        self.project_id, self.project = self.get_project()
        self.headers = {
            'Authorization': f'Token {self.api_key}',
            'Content-Type': 'application/json'
        }

    def get_client(self):
        return Client(url=self.base_url, api_key=self.api_key)

    def get_project(self):
        projects = self.client.get_projects()

        for p in projects:
            if p.get_params().get('title') == self.project_name:
                return p.get_params().get('id'), p

        return None, None
