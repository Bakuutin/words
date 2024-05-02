const { useState, useEffect, useRef } = React

// type Axis = {
//     start: string;
//     end: string;
//     shift: number;
// };


// type Word = {
//     text: string;
//     frequency: number;
//     projection: number[]; // 3D vector
//     distance: number;
// };

AFRAME.registerComponent('look-at-camera', {
    init: function () {
        this.camera = document.querySelector('a-camera').object3D; // Cache the camera's object3D for reuse
        this.lookAtCamera();
    },

    tick: function () {
        this.lookAtCamera();
    },

    lookAtCamera: function () {
        if (this.camera) {
            this.el.object3D.lookAt(this.camera.position); // Use cached camera position
        }
    }
});




const getWords = async () => {
    const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            axes: [
                {
                    start: "water",
                    end: "fire",
                    shift: 1,
                }
            ],
            n: 10,
        }),
    });

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
};

const SearchForm = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [searchResults, setSearchResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!searchTerm) {
            setSearchResults(null);
            return;
        }

        const fetchData = async () => {
            setLoading(true);
            setError(null);

            try {
                const results = await search(searchTerm);
                setSearchResults(results);
            }
            catch (error) {
                setError(error.message);
            }
            setLoading(false);
        };

        fetchData();
    }, [searchTerm]);

    return (
        <form onSubmit={(e) => e.preventDefault()}>
            <input
                type="text"
                placeholder="Search..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
            />
            <button type="submit">Search</button>
            <div>
                {loading && <p>Loading...</p>}
                {error && <p>{error}</p>}
                {searchResults && <pre>{JSON.stringify(searchResults, null, 2)}</pre>}
            </div>
        </form>
    );
}


const Word = ({ word }) => {
    const [x, y, z] = word.projection.map(x => x * 10);

    return (
        <>
            <a-entity size="3" text-geometry={`value: ${word.text}; font: #optimerBoldFont`} look-at-camera position={`${x} ${y} ${z}`}></a-entity>
            {/* <a-text value={word.text} position={`${x} ${y} ${z}`} align="center" color="#ffffff"></a-text> */}
            {/* <a-sphere position={`${x} ${y} ${z}`} radius="0.5" color="red"></a-sphere> */}
        </>
    );
}

const WordCloud = ({ words }) => {
    return (
        <>
            {words.map((word) => (
                <Word word={word} key={word.text} />
            ))}
        </>
    );
}

const body = document.querySelector('body');


let session = null;

body.onclick = () => {
    if (!session) {
        session = true;
        const scene = document.querySelector('a-scene');
        scene.enterVR();
    }
}


const Scene = () => {
    const [words, setWords] = useState([]);

    useEffect(() => {
        async function fetchData() {
            const resp = await getWords();
            setWords(resp.words);
        }

        fetchData().catch(console.error);
    }, []);

    return (
        <a-scene>
            {/* <a-box position="-1 0.5 -3" rotation="0 45 0" color="#4CC3D9"></a-box>
            <a-sphere position="0 1.25 -5" radius="1.25" color="#EF2D5E"></a-sphere>
            <a-cylinder
                position="1 0.75 -3"
                radius="0.5"
                height="1.5"
                color="#FFC65D"
            ></a-cylinder>
            <a-plane
                position="0 0 -4"
                rotation="-90 0 0"
                width="4"
                height="4"
                color="#7BC8A4"
            ></a-plane> */}
            <a-assets>
                <a-asset-item id="optimerBoldFont" src="https://rawgit.com/mrdoob/three.js/dev/examples/fonts/optimer_bold.typeface.json"></a-asset-item>
            </a-assets>


            <WordCloud words={words} />
            {/* <a-torus position="0 0 0" radius="1" radius-tubular="0.0005" color="white" rotation="90 0 0"></a-torus> */}
            <a-camera position="0 0 0"></a-camera>
            <a-entity hand-tracking-controls="hand: left"></a-entity>
            <a-entity hand-tracking-controls="hand: right"></a-entity>
            <a-sky color="#696969"></a-sky>

        </a-scene>
    );
}


ReactDOM.render(<Scene />, document.getElementById('root'));
