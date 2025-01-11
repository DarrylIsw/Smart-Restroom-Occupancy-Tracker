<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class LoginController extends Controller
{
    public function login(Request $request)
    {
        $request->validate([
            'username' => 'required',
            'password' => 'required',
        ]);
    
        // Fetch user by username
        $user = User::where('username', $request->username)->first();
    
        if (!$user || $user->password !== $request->password) {
            // Return error response for invalid credentials
            return response()->json(['message' => 'Invalid username or password'], 401);
        }
    
        // Return success response with role
        return response()->json([
            'message' => 'Login successful',
            'role' => $user->role,
        ], 200);
    }

    public function logout(Request $request)
    {
        // Start session if not started
        if (!session()->isStarted()) {
            session()->start();
        }

        // Invalidate the session
        session()->invalidate();

        // Regenerate the CSRF token to prevent reuse
        session()->regenerateToken();

        return response()->json([
            'message' => 'Successfully logged out.',
        ], 200);
    }
    

    public function authenticate(Request $request)
    {
        $username = $request->input('username');
        $password = $request->input('password');

        // Basic user credential check
        $user = DB::table('users')->where('username', $username)->first();

        if ($user && $user->password === $password) {
            // Store user info in session
            $request->session()->put('user_id', $user->id);
            $request->session()->put('username', $user->username);
            $request->session()->put('role', $user->role);

            // Redirect to home if role is admin or janitor
            if ($user->role === 'admin' || $user->role === 'janitor') {
                return redirect()->route('home');
            }
        }

        return back()->withErrors([
            'error' => 'The provided credentials do not match our records.',
        ]);
    }
    /**
     * Display a listing of the resource.
     */
    public function index()
    {
        //
    }

    /**
     * Store a newly created resource in storage.
     */
    public function store(Request $request)
    {
        //
    }

    /**
     * Display the specified resource.
     */
    public function show(string $id)
    {
        //
    }

    /**
     * Update the specified resource in storage.
     */
    public function update(Request $request, string $id)
    {
        //
    }

    /**
     * Remove the specified resource from storage.
     */
    public function destroy(string $id)
    {
        //
    }
}
