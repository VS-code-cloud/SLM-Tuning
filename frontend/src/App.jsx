import Hero from './components/Hero.jsx'
import UseCases from './components/UseCases.jsx'
import HowItWorks from './components/HowItWorks.jsx'
import Compare from './components/Compare.jsx'
import Footer from './components/Footer.jsx'

export default function App() {
  return (
    <div className="page">
      <Hero />
      <main>
        <UseCases />
        <HowItWorks />
        <Compare />
      </main>
      <Footer />
    </div>
  )
}
