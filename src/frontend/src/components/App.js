import React, { Component } from 'react';
import { TryOn } from './TryOn';
// Importer seulement la fonction pour changer les lunettes
import { changeGlassesModel } from './render.js'; 
// Importer le nouveau fichier CSS
import '../style/App.css';

// Liste statique de tous les modèles disponibles
const ALL_GLASSES_MODELS = [
  { id: "purple1", name: "Purple Style 1", category: "Mode" },
  { id: "red", name: "Red Classic", category: "Classique" },
  { id: "blue_aviator", name: "Blue Aviator", category: "Sport" },
  { id: "black_round", name: "Black Round", category: "Rétro" },
  // Ajoutez d'autres modèles ici selon vos fichiers dans public/obj/
];

export default class App extends Component {

  constructor(props) {
    super(props);
    this.state = {
      analysisResult: null,
      recommendedGlassesIds: [],
      currentDisplayModelId: "purple1", // Modèle initial
      error: null,
      isLoadingAnalysis: false,
      categories: this.getUniqueCategories(),
      activeCategory: 'Tous'
    };
    this.handleAnalyseClick = this.handleAnalyseClick.bind(this);
    this.handleSelectGlasses = this.handleSelectGlasses.bind(this);
  }

  componentDidMount() {
    // Assurez-vous que le modèle initial est chargé au démarrage si nécessaire
    // (IntializeThreejs le fait déjà, mais c'est pour la cohérence)
    if (this.state.currentDisplayModelId) {
      // Note: On ne peut pas appeler changeGlassesModel directement ici
      // car Three.js n'est peut-être pas encore prêt.
      // IntializeThreejs s'en charge.
    }
  }

  // Extraire les catégories uniques
  getUniqueCategories = () => {
    const allCategories = ALL_GLASSES_MODELS.map(model => model.category);
    return ['Tous', ...new Set(allCategories)];
  }

  // Filtrer les modèles par catégorie
  filterModelsByCategory = (models, category) => {
    if (category === 'Tous') return models;
    return models.filter(model => model.category === category);
  }

  handleCategoryChange = (category) => {
    this.setState({ activeCategory: category });
  }

  handleAnalyseClick() {
    this.setState({ 
      isLoadingAnalysis: true, 
      error: null, 
      analysisResult: null, 
      recommendedGlassesIds: [] 
    });

    // Données factices pour la démonstration
    const dummyLandmarks = Array(468).fill([0, 0, 0]);

    fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks: dummyLandmarks })
    })
      .then(response => {
        if (!response.ok) {
          return response.json().then(errData => {
             throw new Error(`HTTP error ${response.status}: ${errData.detail || response.statusText}`);
          }).catch(() => {
            throw new Error(`HTTP error! status: ${response.status} ${response.statusText}`);
          });
        }
        return response.json();
      })
      .then(data => {
        this.setState({
          analysisResult: data.predicted_shape || 'Forme non déterminée.',
          recommendedGlassesIds: data.recommended_glasses || [],
          isLoadingAnalysis: false,
        });
      })
      .catch(error => {
        console.error("Erreur lors de l'appel à /predict:", error);
        this.setState({ 
          error: `Erreur lors de l'analyse : ${error.message}`, 
          isLoadingAnalysis: false 
        });
      });
  }

  // Fonction appelée lorsqu'un bouton de lunettes est cliqué
  handleSelectGlasses(modelId) {
    if (modelId && modelId !== this.state.currentDisplayModelId) {
      console.log("Sélection des lunettes:", modelId);
      
      changeGlassesModel(modelId);
      
      this.setState({ currentDisplayModelId: modelId });
    }
  }

  renderGlassesButton = (model) => {
    const { currentDisplayModelId } = this.state;
    return (
      <button 
        key={model.id} 
        className={`button glasses-button ${model.id === currentDisplayModelId ? 'button--active' : ''}`}
        onClick={() => this.handleSelectGlasses(model.id)}
      >
        <span className="glasses-button__name">{model.name}</span>
        {model.category && (
          <span className="glasses-button__category">{model.category}</span>
        )}
      </button>
    );
  }

  renderCategoryTabs = () => {
    const { categories, activeCategory } = this.state;
    
    return (
      <div className="category-tabs">
        {categories.map(category => (
          <button
            key={category}
            className={`category-tab ${activeCategory === category ? 'category-tab--active' : ''}`}
            onClick={() => this.handleCategoryChange(category)}
          >
            {category}
          </button>
        ))}
      </div>
    );
  }

  render() {
    const { 
      analysisResult, 
      recommendedGlassesIds, 
      error, 
      isLoadingAnalysis,
      activeCategory 
    } = this.state;

    // Filtrer les lunettes par catégorie active
    const filteredModels = this.filterModelsByCategory(ALL_GLASSES_MODELS, activeCategory);

    // Trouver les objets complets des lunettes recommandées
    const recommendedGlassesObjects = ALL_GLASSES_MODELS.filter(model => 
        recommendedGlassesIds.includes(model.id)
    );

    // Pour l'instant, on affiche seulement 2 recommandations maximum
    const limitedRecommendations = recommendedGlassesObjects.slice(0, 2);

    return (
      <div className="app-container">
        <header className="app-header">
          <h1 className="app-title">Optical Factory </h1>
          <p className="app-tagline">Essayez virtuellement nos lunettes pour trouver votre style parfait</p>
        </header>
        
        {/* Section Essai Virtuel */}
        <section className="tryon-section">
          <TryOn /> 
        </section>

        {/* Section Contrôles et Analyse */}
        <section className="controls-section">
          <h2 className="section-title">Analyser votre visage</h2>
          <p className="section-description">
            Lancez l'analyse pour obtenir des recommandations basées sur la forme de votre visage.
          </p>
          
          <button 
            className={`button button--primary button--fullwidth ${isLoadingAnalysis ? 'button--loading' : ''}`}
            onClick={this.handleAnalyseClick} 
            disabled={isLoadingAnalysis}
          >
            {isLoadingAnalysis ? (
              <span className="button__spinner"></span>
            ) : (
              'Lancer l\'analyse du visage'
            )}
          </button>
          
          {error && (
            <div className="alert alert--error">{error}</div>
          )}
          
          {analysisResult && !isLoadingAnalysis && (
             <div className="analysis-result">
               <h3 className="analysis-result__title">Résultat de l'analyse</h3>
               <p className="analysis-result__text">Forme de visage détectée : <strong>{analysisResult}</strong></p>
             </div>
          )}
        </section>

        {/* Section Recommandations */}
        {!isLoadingAnalysis && recommendedGlassesIds.length > 0 && (
          <section className="glasses-section recommendations-section">
            <h3 className="section-title">Recommandations personnalisées</h3>
            <p className="section-description">
              Ces modèles sont sélectionnés spécifiquement pour la forme de votre visage.
            </p>
            
            <div className="glasses-grid">
              {limitedRecommendations.map(this.renderGlassesButton)}
            </div>
          </section>
        )}

        {/* Section Tous les Modèles */}
        <section className="glasses-section all-models-section">
          <h3 className="section-title">Tous nos modèles</h3>
          <p className="section-description">
            Explorez notre collection complète et trouvez le style qui vous correspond.
          </p>
          
          {this.renderCategoryTabs()}
          
          <div className="glasses-grid glasses-grid--all">
            {filteredModels.map(this.renderGlassesButton)}
          </div>
        </section>
        
        <footer className="app-footer">
          <p>© {new Date().getFullYear()} Optical Factory  - Application d'essayage virtuel de lunettes</p>
        </footer>
      </div>
    );
  }
}
