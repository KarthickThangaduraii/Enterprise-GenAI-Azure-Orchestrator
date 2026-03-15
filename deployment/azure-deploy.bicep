@description('The name of the Azure OpenAI service.')
param openAiServiceName string = 'openai-${uniqueString(resourceGroup().id)}'

@description('The location for the resources.')
param location string = resourceGroup().location

@description('The name of the Azure AI Search service.')
param searchServiceName string = 'search-${uniqueString(resourceGroup().id)}'

@description('The pricing tier for the Azure AI Search service.')
param searchServiceSku string = 'standard'

resource openAiService 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAiServiceName
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAiServiceName
    publicNetworkAccess: 'Enabled'
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAiService
  name: 'text-embedding-ada-002'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-ada-002'
      version: '2'
    }
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAiService
  name: 'gpt-4'
  sku: {
    name: 'Standard'
    capacity: 10
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4'
      version: 'turbo-2024-04-09'
    }
  }
}

resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchServiceName
  location: location
  sku: {
    name: searchServiceSku
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
  }
}

output openAiEndpoint string = openAiService.properties.endpoint
output searchEndpoint string = 'https://${searchServiceName}.search.windows.net'
