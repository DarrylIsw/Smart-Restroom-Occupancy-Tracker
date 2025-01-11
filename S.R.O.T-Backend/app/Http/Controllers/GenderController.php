<?php

namespace App\Http\Controllers;

use App\Models\Gender;
use Illuminate\Http\Request;

class GenderController extends Controller
{
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        $genders = Gender::all();
        return response()->json($genders);
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        $validated = $request->validate([
            'label' => 'required|string|unique:genders,label',
        ]);

        $gender = Gender::create($validated);

        return response()->json($gender, 201);
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        $gender = Gender::findOrFail($id);
        return response()->json($gender);
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        $gender = Gender::findOrFail($id);

        $validated = $request->validate([
            'label' => 'string|unique:genders,label,' . $gender->id,
        ]);

        $gender->update($validated);

        return response()->json($gender);
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        $gender = Gender::findOrFail($id);
        $gender->delete();

        return response()->json(['message' => 'Gender deleted successfully']);
    }
}
